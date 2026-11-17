import pandas as pd
import numpy as np
import time
import pulp
import warnings

warnings.filterwarnings("ignore")

# ======================================================================
#  SECTION 1 — CONFIGURATION
# ======================================================================

XLSX = "Food_Data_with_Fats_Completed_v3.xlsx"

SHEET_CAT = {
    "Vegetables (4)": "V",
    "Rice & pasta (4)": "C",
    "Nuts & Seeds (4)": "S",
    "Meats (4)": "P",
    "Fruits (4)": "F",
    "Fats & Oils (4)": "Fat",
    "Dairy___Eggs__3": None,
}

MEALS = ["Breakfast", "Lunch", "Dinner", "Snack"]

MEAL_SLOTS = {
    "Breakfast": [
        ("P",   True,  80, 130),   # slot 0 — egg
        ("D",   True, 120, 180),   # slot 1 — dairy
        ("F",   True, 100, 300),   # slot 2 — fruit
        ("C",   False, 60,  80),   # slot 3 — oats/carb
        ("S",   False, 15,  25),   # slot 4 — seed
    ],
    "Lunch": [
        ("P",   True,  150, 300),  # slot 0 — protein
        ("V",   True,   80, 150),  # slot 1 — vegetable
        ("C",   True,  100, 500),  # slot 2 — carb
        ("Fat", False,   5,  15),  # slot 3 — fat/oil
    ],
    "Dinner": [
        ("P",   True,  150, 200),  # slot 0 — protein
        ("V",   True,   80, 150),  # slot 1 — vegetable
        ("C",   True,  150, 500),  # slot 2 — carb
    ],
    "Snack": [
        ("F",   True,  100, 300),  # slot 0 — fruit
        ("S",   False,  15,  25),  # slot 1 — seed
    ],
}

# Randomly-Activated Slots RS = {(Breakfast,0), (Breakfast,3)}
# Each entry: (meal, slot_idx, p_s)  where p_s is the Bernoulli probability.
# Bernoulli(p_s) is drawn BEFORE each day's LP is built (C20).
RANDOM_SLOTS = [
    ("Breakfast", 0, 0.60),   # egg  slot — appears on ~60 % of days
    ("Breakfast", 3, 0.60),   # oats slot — appears on ~60 % of days
]
# Convenience set for quick membership tests
RS_SET = {(m, si) for m, si, _ in RANDOM_SLOTS}

# Slots whose gram bounds are NOT scaled by user.portion_scale (§2.3 NO_SCALE)
NO_SCALE_SLOTS = {("Breakfast", 0), ("Breakfast", 1), ("Breakfast", 3)}

# Per-food gram overrides — caps foods that are nutritionally dense
# Applies BEFORE the LP is built; overrides Portion_max for that food only.
MAX_FOOD_GRAMS_OVERRIDE = {
    "coconut meat dried": 30,
    "coconut meat":       120,
    "dates deglet":       100,
    "dates medjool":       80,
    "olive black":        100,
    "olive green":        100,
}

FAT_FILTER = {
        "P":   0.20,
        "D":   0.28,
        "F":   0.20,
        "V":   0.35,
        "C":   0.15,
}

FAMILY_PLAN_LIMITS = {
    "poultry":       8,
    "red_meat_fatty": 4,
    "red_meat_lean":  4,
    "offal":          1,
    "fish_fatty":     1,
    "fish_lean":      1,
    "seafood":        1,
}

# Tracked slots for cross-day food exclusion
VARY_ACROSS_DAYS = {
    "P":   [("Lunch", 0), ("Dinner", 0), ("Breakfast", 0)],
    "C":   [("Lunch", 2), ("Dinner", 2), ("Breakfast", 3)],
    "V":   [("Lunch", 1), ("Dinner", 1)],
    "F":   [("Breakfast", 2), ("Snack", 0)],
    "Fat": [("Lunch", 3)],
    "D":   [("Breakfast", 1)],
}

# §2.5b  MIN_ALT: minimum alternatives that must remain before banning a food
MIN_ALTERNATIVES_TO_BAN = 5   # = MIN_ALT in the model document

MEAL_CAL_FRAC = {
    "Breakfast": 0.25,
    "Lunch":     0.35,
    "Dinner":    0.28,
    "Snack":     0.12,
}
MEAL_CAL_TOL    = 0.15   # +15% and -15% per-meal calorie band
NETCARB_MEAL_TOL = 0.30  # +30 % per-meal NC tolerance

GP_WEIGHTS_BY_GOAL = {
    "lose":     {"cost": 0.5},
    "maintain": {"cost": 0.5},
    "gain":     {"cost": 0.5},
}

# Diabetes / Weight for diabetes / Weight for unhealthy people (sat fat focus)
DIABETES_PARAMS = {
    "none":       {"nc_max_day": None, "w_diab": 0.0,  "w_health": 0.0},
    "prediabetic":{"nc_max_day": 150,  "w_diab": 2.0,  "w_health": 1.0},
    "type2":      {"nc_max_day": 130,  "w_diab": 5.0,  "w_health": 1.0},
}

# Cardiovascular / Heart-Healthy mode
# This activates w_health = 1.0, identical to the diabetes sat-fat penalty.
HYPERTENSION_PARAMS = {
    "none":           {"w_health": 0.0},
    "cardiovascular": {"w_health": 1.0},
}

ROUNDING_TOL = 0.01   # 1 % — applied to all macro ceiling checks

MACRO_SPLITS = {
    "lose":     (0.25, 0.40),
    "maintain": (0.30, 0.40),
    "gain":     (0.25, 0.50),
}

BASE_CAL = 2000

ALLERGY_COLS = {
    "lactose": "Lactose Allergy",
    "nuts":    "Nuts Allergy",
    "gluten":  "Gluten Allergy",
}


# ======================================================================
#  SECTION 2 — DATA LOADING
# ======================================================================

def load_foods(allergies=None):
    allergies = [a.lower() for a in (allergies or [])]
    xl = pd.ExcelFile(XLSX)
    frames = []

    for sheet in xl.sheet_names:
        if sheet not in SHEET_CAT:
            continue
        df = xl.parse(sheet)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={df.columns[0]: "Food"})
        df["Food"] = df["Food"].astype(str).str.strip()
        df = df[df["Food"].notna() & (df["Food"] != "") & (df["Food"] != "nan")].copy()

        if SHEET_CAT[sheet] is not None:
            df["Category"] = SHEET_CAT[sheet]
        else:
            if "Category" not in df.columns:
                df["Category"] = "D"

        for allergy in allergies:
            col = ALLERGY_COLS.get(allergy)
            if col and col in df.columns:
                df = df[df[col] != 1]

        MAP = {
            "Calories (kCal)":      "cal",
            "Protein (g)":          "prot",
            "Fats (g)":             "fat",
            "Saturated (g)":        "sat_fat",
            "Carbohydrates (g)":    "carbs",
            "Fiber (g)":            "fiber",
            "Net Carbs (g)":        "net_carbs",
            "Cost":                 "cost",
        }
        for src, dst in MAP.items():
            col_data = df[src] if src in df.columns else pd.Series(0, index=df.index)
            df[dst] = pd.to_numeric(col_data, errors="coerce").fillna(0)

        if "Net Carbs (g)" not in df.columns:
            df["net_carbs"] = (df["carbs"] - df["fiber"]).clip(lower=0)

        if "Food_Family" in df.columns:
            df["Food_Family"] = df["Food_Family"].astype(str).str.strip()
        else:
            df["Food_Family"] = "other"

        for meal in MEALS:
            if meal not in df.columns:
                df[meal] = 0
            df[meal] = pd.to_numeric(df[meal], errors="coerce").fillna(0).astype(int)

        if sheet == "Fats & Oils (4)":
            for meal in ["Breakfast", "Lunch", "Dinner"]:
                df[meal] = 1
            df["Snack"] = 0

        frames.append(df[["Food", "Category", "Food_Family",
                           "cal", "prot", "fat", "sat_fat",
                           "carbs", "fiber", "net_carbs", "cost"] + MEALS])

    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


# ======================================================================
#  SECTION 3 — SLOT TRIPLES  (incorporates C20 Bernoulli draw)
# ======================================================================

def build_slot_triples(foods, excluded=None, family_banned=None, user=None, rng=None):
    """
    Build (food_idx, meal, slot_idx, min_g, max_g, required) triples for one day.
        Foods in excluded[(meal, slot_idx)] are omitted from the eligible set for
        that slot.  The safety check (MIN_ALT) is applied in build_exclusions after
        the solve.
    """
    excluded     = excluded     or {}
    family_banned = family_banned or set()
    scale        = user.portion_scale if user is not None else 1.0

    # seed is already set on rng before this call (master_seed + day + slot_off)
    # random slot activation for eggs/oats
    active_random_slots: set = set()
    for meal, s_idx, p_s in RANDOM_SLOTS:
        delta = 1 if (rng is None or rng.random() < p_s) else 0  # Bernoulli(p_s)
        if delta == 1:
            active_random_slots.add((meal, s_idx))

    triples = []

    for meal, slots in MEAL_SLOTS.items():
        for s_idx, (cat, required, mn, mx) in enumerate(slots):

            # if the active slot isn't activated, skip that slot
            if (meal, s_idx) in RS_SET:
                if (meal, s_idx) not in active_random_slots:
                    continue

            # Portion scaling (NO_SCALE_SLOTS are exempt — §2.3)
            if cat in ("S", "Fat") or (meal, s_idx) in NO_SCALE_SLOTS:
                scaled_mn, scaled_mx = mn, mx
            else:
                scaled_mn = round(mn * scale / 5) * 5
                scaled_mx = round(mx * scale / 5) * 5

            # Cross-day exclusion set
            banned = excluded.get((meal, s_idx), set())

            fat_thresh = FAT_FILTER.get(cat, 99)

            eligible = foods[
                (foods["Category"] == cat) &
                (foods[meal] == 1) &
                (foods["fat"] <= fat_thresh) &
                (~foods.index.isin(banned)) &
                (~foods["Food_Family"].isin(family_banned))
            ].index

            for j in eligible:
                food_name = str(foods.loc[j, "Food"]).lower()
                effective_mx = scaled_mx
                for substr, cap in MAX_FOOD_GRAMS_OVERRIDE.items():
                    if substr in food_name:
                        effective_mx = min(scaled_mx, cap)
                        break
                triples.append((j, meal, s_idx, scaled_mn, effective_mx, required))

    return triples


# ======================================================================
#  SECTION 4 — USER BIOMETRICS
# ======================================================================

ACTIVITY_MULTIPLIERS = {
    "sedentary":   1.2,
    "light":       1.375,
    "moderate":    1.55,
    "active":      1.725,
    "very_active": 1.9,
}
GOAL_DELTA = {"lose": -500, "maintain": 0, "gain": 500}


class UserBiometrics:
    # Computes all personalized targets from biometrics

    def __init__(self, age, gender, height_cm, weight_kg,
                 activity_level="moderate", goal="maintain",
                 diabetes_status="none", hypertension_status="none",
                 allergies=None, budget=None):

        self.age           = age
        self.gender        = gender.lower()
        self.height_cm     = height_cm
        self.weight_kg     = weight_kg
        self.activity_level = activity_level.lower()
        self.goal          = goal.lower()

        # Normalise diabetes status string
        ds = diabetes_status.lower().replace(" ", "").replace("-", "")
        if ds in ("typeii", "type2diabetes", "t2"):
            ds = "type2"
        elif ds in ("pre", "pre-diabetes", "prediabetes"):
            ds = "prediabetic"
        self.diabetes_status = ds

        self.allergies = [a.lower() for a in (allergies or [])]
        self.budget    = budget

        # diabetes-derived weights and NC cap
        diab = DIABETES_PARAMS.get(self.diabetes_status, DIABETES_PARAMS["none"])
        self.nc_max_day = diab["nc_max_day"]
        self.w_diab     = diab["w_diab"]
        self.w_health   = diab["w_health"]

        # Cardiovascular mode (sat-fat reduction only)
        hs = hypertension_status.lower().replace(" ", "").replace("-", "")
        if hs in ("high", "hypertension", "htn", "yes", "true", "1",
                  "standard", "strict", "cardiovascular"):
            hs = "cardiovascular"
        else:
            hs = "none"
        self.hypertension_status = hs

        htn = HYPERTENSION_PARAMS[hs]
        # w_health is activated if EITHER diabetes OR cardiovascular mode is on
        if htn["w_health"] > 0.0:
            self.w_health = max(self.w_health, htn["w_health"])

        # Calories
        if self.gender == "male":
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

        mult            = ACTIVITY_MULTIPLIERS.get(self.activity_level, 1.55)
        self.tdee       = bmr * mult
        self.cal_target = self.tdee + GOAL_DELTA.get(self.goal, 0)

        # Macro targets
        prot_per_kg          = 2.0 if goal == "gain" else 1.6
        self.protein_target  = round(weight_kg * prot_per_kg, 1)

        fat_pct, carb_pct    = MACRO_SPLITS.get(self.goal, (0.27, 0.40))
        self.fat_pct         = fat_pct
        self.carb_pct        = carb_pct
        self.fat_target      = round(self.cal_target * fat_pct / 9, 1)
        self.carbs_target    = round(self.cal_target * carb_pct / 4, 1)

        self.fiber_target    = 38 if self.gender == "male" else 25

        # Macro upper bounds
        self.protein_max = round(weight_kg * 2.5, 1)
        self.fiber_max = self.fiber_target * 2

        self.portion_scale   = max(1.0, self.cal_target / BASE_CAL)
        self.meal_size_max   = max(700, round(self.cal_target * 0.40))

    @property
    def bmi(self):
        h = self.height_cm / 100
        return round(self.weight_kg / (h * h), 1)

    @property
    def bmi_category(self):
        b = self.bmi
        if b < 18.5: return "Underweight"
        if b < 25.0: return "Normal weight"
        if b < 30.0: return "Overweight"
        return "Obese"

    def summary(self):
        act = self.activity_level.replace("_", " ").title()
        return "\n".join([
            f"\n  {'_' * 60}",
            f"  USER",
            f"  {'_' * 60}",
            f"  {self.age} yrs  |  {self.gender.title()}  |  "
            f"{self.height_cm} cm  |  {self.weight_kg} kg  |  BMI {self.bmi} ({self.bmi_category})",
            f"  Activity: {act}  |  Goal: {self.goal.upper()}",
            f"  Diabetes: {self.diabetes_status}  |  W_diab: {self.w_diab}  |  W_health: {self.w_health}"
            + (f"  |  NC_max/day: {self.nc_max_day} g" if self.nc_max_day else "  |  NC_max/day: none"),
            f"  Cardiovascular mode: {self.hypertension_status}"
            + ("  (sat-fat minimized via W_health)" if self.hypertension_status != "none" else ""),
            *([f"  Allergies: {', '.join(self.allergies)}"] if self.allergies else []),
            f"  {'_' * 60}",
            f"  DAILY TARGETS",
            f"  {'_' * 60}",
            f"  Calories : {self.cal_target:.0f} kcal  (TDEE {self.tdee:.0f})",
            f"  Protein  : {self.protein_target} – {self.protein_max} g  "
            f"({'2.0' if self.goal == 'gain' else '1.6'}–2.5 g/kg)",
            f"  Fat      : <= {self.fat_target} g  ({self.fat_pct * 100:.0f}% of kcal)",
            f"  Carbs    : <= {self.carbs_target} g  ({self.carb_pct * 100:.0f}% of kcal)",
            f"  Fiber    : {self.fiber_target} – {self.fiber_max} g",
            *([f"  Budget   : {self.budget} EGP/day"] if self.budget else []),
            f"  {'_' * 60}",
        ])


# ======================================================================
#  SECTION 5 — MILP MODEL  (single day)
# ======================================================================

def _compute_reference_values(user, foods, triples):
    # Compute R_cost, R_SF, R_NC for Normalization.

    # R_cost: sum over every (meal, slot) of [max cost food × that slot's Portion_max]
    # to get the worst case cost
    r_cost = 0.0
    seen_slots = set()
    for j, meal, s_idx, mn, mx, req in triples:
        key = (meal, s_idx)
        if key in seen_slots:
            continue
        seen_slots.add(key)
        slot_foods = [
            foods.loc[jj, "cost"] * float(mx_)
            for jj, m2, s2, mn_, mx_, _ in triples
            if m2 == meal and s2 == s_idx
        ]
        if slot_foods:
            r_cost += max(slot_foods)

    # Safety guard: avoid division by zero if foods table has zero costs
    r_cost = max(r_cost, 1.0)

    # R_SF = 10 % of energy ÷ 9 kcal/g  →  grams of saturated fat
    r_sf = max(0.10 * user.cal_target / 9.0, 1.0)

    # R_NC: use diabetes NC ceiling for diabetic users, carb ceiling otherwise
    if user.nc_max_day is not None:
        r_nc = float(user.nc_max_day)
    else:
        r_nc = float(user.carbs_target)
    r_nc = max(r_nc, 1.0)

    return r_cost, r_sf, r_nc




def solve_one_day(user, foods, triples, day_label="", seed=None, perturb_cost=True):
    # Build and solve the MILP for one day using the normalized objective.

    rng = np.random.default_rng(seed)

    perturbed_cost = {}
    if perturb_cost:
        PERTURB_CATS = {"P": 1.00, "V": 0.50, "F": 0.50, "C": 1.00}
        for cat, pscale in PERTURB_CATS.items():
            cat_foods = foods[foods["Category"] == cat]
            if cat_foods.empty:
                continue
            max_cost = float(cat_foods["cost"].max())
            for j in cat_foods.index:
                noise = rng.uniform(0, max_cost * pscale)
                perturbed_cost[j] = float(foods.loc[j, "cost"]) + noise

    prob = pulp.LpProblem(f"Diet_{day_label}", pulp.LpMinimize)

    # Decision variables
    x, y = {}, {}
    for j, meal, s_idx, mn, mx, req in triples:
        k = (j, meal, s_idx)
        x[k] = pulp.LpVariable(f"x_{j}_{meal}_{s_idx}", lowBound=0, upBound=float(mx))
        y[k] = pulp.LpVariable(f"y_{j}_{meal}_{s_idx}", cat="Binary")

    # Compute normalization reference values
    R_cost, R_SF, R_NC = _compute_reference_values(user, foods, triples)

    # User Activity Level Multiplier
    W = GP_WEIGHTS_BY_GOAL.get(user.goal, GP_WEIGHTS_BY_GOAL["maintain"])

    # Normalizations
    # Cost fraction of worst-case feasible cost
    f1_cost = pulp.lpSum(
        perturbed_cost.get(j, float(foods.loc[j, "cost"])) * x[j, m, si]
        for j, m, si, *_ in triples
    )

    # Saturated fat fraction of SatFat limit
    f3_satfat = pulp.lpSum(
        foods.loc[j, "sat_fat"] * x[j, m, si]
        for j, m, si, *_ in triples
    )

    # Net carb fraction of the diabetes/carb ceiling
    f4_netcarb = pulp.lpSum(
        foods.loc[j, "net_carbs"] * x[j, m, si]
        for j, m, si, *_ in triples
    )

    # Normalized objective
    # cost term + sat-fat term (diabetes/cardiovascular) + net-carb term (diabetes only)
    prob += (
        W["cost"]        * (f1_cost    / R_cost)
        + user.w_health  * (f3_satfat  / R_SF)
        + user.w_diab    * (f4_netcarb / R_NC)
    ), "Obj_Normalized"

    # Utility helpers
    def sum_all(nut):
        return pulp.lpSum(foods.loc[j, nut] * x[j, m, si]
                          for j, m, si, *_ in triples)

    def sum_n(meal_name, nut):
        return pulp.lpSum(foods.loc[j, nut] * x[j, m, si]
                          for j, m, si, *_ in triples if m == meal_name)

    prob += sum_all("cal") >= user.cal_target * 0.97
    prob += sum_all("cal") <= user.cal_target * 1.03

    prob += sum_all("prot")  >= user.protein_target
    prob += sum_all("fat")   <= user.fat_target
    prob += sum_all("carbs") <= user.carbs_target
    prob += sum_all("fiber") >= user.fiber_target

    # Protein daily ceiling
    # Prevents the solver from stuffing every protein slot to its gram maximum
    prob += sum_all("prot") <= user.protein_max

    # Fiber daily ceiling
    prob += sum_all("fiber") <= user.fiber_max

    # Portion bounds linking x to y (Big-M)
    for j, meal, s_idx, mn, mx, req in triples:
        k = (j, meal, s_idx)
        prob += x[k] <= float(mx) * y[k]
        prob += x[k] >= float(mn) * y[k]

    # Slot selection: required =1, optional ≤1
    for meal, slots in MEAL_SLOTS.items():
        for s_idx, (cat, required, mn, mx) in enumerate(slots):
            sy = [y[j, m, si] for j, m, si, *_ in triples
                  if m == meal and si == s_idx]
            if not sy:
                continue

            if (meal, s_idx) in RS_SET:
                prob += pulp.lpSum(sy) <= 1
            elif required:
                prob += pulp.lpSum(sy) == 1
            else:
                prob += pulp.lpSum(sy) <= 1

    # Meal physical size cap
    MEAL_GRAM_CAPS = {
        "Breakfast": 650,
        "Lunch":     900,
        "Dinner":    900,
        "Snack":     300,
    }
    for meal in MEALS:
        cap = min(user.meal_size_max, MEAL_GRAM_CAPS[meal])
        prob += (pulp.lpSum(x[j, m, si] for j, m, si, *_ in triples if m == meal)
                 <= cap), f"C13_size_{meal}"

    # Each food appears in at most 1 slot per day
    for j in foods.index:
        jy = [y[j2, m, si] for j2, m, si, *_ in triples if j2 == j]
        if len(jy) > 1:
            prob += pulp.lpSum(jy) <= 1

    # Different protein & carb between Lunch and Dinner
    lunch_slots  = {cat: si for si, (cat, *_) in enumerate(MEAL_SLOTS["Lunch"])}
    dinner_slots = {cat: si for si, (cat, *_) in enumerate(MEAL_SLOTS["Dinner"])}
    for cat in ["P", "C"]:
        l_si = lunch_slots.get(cat)
        d_si = dinner_slots.get(cat)
        if l_si is None or d_si is None:
            continue
        for j in foods[foods["Category"] == cat].index:
            ly = y.get((j, "Lunch",  l_si))
            dy = y.get((j, "Dinner", d_si))
            if ly is not None and dy is not None:
                prob += ly + dy <= 1

    # Optional budget constraint
    if user.budget is not None:
        prob += (pulp.lpSum(foods.loc[j, "cost"] * x[j, m, si]
                            for j, m, si, *_ in triples)
                 <= user.budget)

    # Diabetes net-carb caps
    if user.nc_max_day is not None:
        prob += sum_all("net_carbs") <= user.nc_max_day
        for meal in MEALS:
            meal_cap = user.nc_max_day * MEAL_CAL_FRAC[meal] * (1 + NETCARB_MEAL_TOL)
            prob += sum_n(meal, "net_carbs") <= meal_cap

    # Calorie ordering Lunch ≥ Dinner ≥ Breakfast ≥ Snack
    prob += sum_n("Lunch",     "cal") >= sum_n("Dinner",    "cal")
    prob += sum_n("Dinner",    "cal") >= sum_n("Breakfast", "cal")
    prob += sum_n("Breakfast", "cal") >= sum_n("Snack",     "cal")

    # Per-meal calorie band
    for meal in MEALS:
        tgt = user.cal_target * MEAL_CAL_FRAC[meal]
        prob += sum_n(meal, "cal") <= tgt * (1 + MEAL_CAL_TOL)
        prob += sum_n(meal, "cal") >= tgt * (1 - MEAL_CAL_TOL)

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120, gapRel=0.02)
    status = prob.solve(solver)
    sol    = pulp.LpStatus[status]

    if pulp.value(prob.objective) is None:
        return None, sol

    plan = {m: [] for m in MEALS}
    for j, meal, s_idx, mn, mx, req in triples:
        k = (j, meal, s_idx)
        if pulp.value(y[k]) is None or pulp.value(y[k]) < 0.5:
            continue
        grams = pulp.value(x[k])
        if grams is None or grams < 10:
            continue
        g = max(round(grams / 5) * 5, 10)
        plan[meal].append({
            "food":        foods.loc[j, "Food"],
            "food_idx":    j,
            "category":    foods.loc[j, "Category"],
            "food_family": foods.loc[j, "Food_Family"],
            "slot":        s_idx,
            "grams":       g,
            "calories":    round(g * foods.loc[j, "cal"],      1),
            "protein":     round(g * foods.loc[j, "prot"],     1),
            "fat":         round(g * foods.loc[j, "fat"],      1),
            "sat_fat":     round(g * foods.loc[j, "sat_fat"],  1),
            "carbs":       round(g * foods.loc[j, "carbs"],    1),
            "fiber":       round(g * foods.loc[j, "fiber"],    1),
            "net_carbs":   round(g * foods.loc[j, "net_carbs"],1),
            "cost":        round(g * foods.loc[j, "cost"],     2),
        })

    # Post-rounding micro-adjustment
    def _recompute(plan_, nut):
        return sum(it[nut] for m in MEALS for it in plan_.get(m, []))

    def _trim_for(plan_, nut, ceiling, foods_):
        for _ in range(20):
            if _recompute(plan_, nut) <= ceiling:
                break
            best_meal, best_idx, best_val = None, None, 0.0
            for m in MEALS:
                for idx, it in enumerate(plan_.get(m, [])):
                    if it[nut] > best_val and it["grams"] > 10:
                        best_meal, best_idx, best_val = m, idx, it[nut]
            if best_meal is None:
                break
            it  = plan_[best_meal][best_idx]
            j_  = it["food_idx"]
            ng  = max(it["grams"] - 5, 10)
            it["grams"]     = ng
            it["calories"]  = round(ng * foods_.loc[j_, "cal"],       1)
            it["protein"]   = round(ng * foods_.loc[j_, "prot"],      1)
            it["fat"]       = round(ng * foods_.loc[j_, "fat"],       1)
            it["sat_fat"]   = round(ng * foods_.loc[j_, "sat_fat"],   1)
            it["carbs"]     = round(ng * foods_.loc[j_, "carbs"],     1)
            it["fiber"]     = round(ng * foods_.loc[j_, "fiber"],     1)
            it["net_carbs"] = round(ng * foods_.loc[j_, "net_carbs"], 1)
            it["cost"]      = round(ng * foods_.loc[j_, "cost"],      2)

    _trim_for(plan, "fat",   user.fat_target,   foods)
    _trim_for(plan, "carbs", user.carbs_target, foods)
    _trim_for(plan, "fiber", user.fiber_max,    foods)
    if user.nc_max_day is not None:
        _trim_for(plan, "net_carbs", user.nc_max_day, foods)

    # Calorie boost
    cal_floor = user.cal_target * 0.97
    for _ in range(60):
        total_cal = _recompute(plan, "calories")
        if total_cal >= cal_floor:
            break

        # Find the item with highest kcal/g that can be increased without
        # pushing fat, carbs, or protein over their respective ceilings
        best_meal, best_idx, best_density = None, None, 0.0
        for m in MEALS:
            for idx, it in enumerate(plan.get(m, [])):
                j_ = it["food_idx"]
                density = float(foods.loc[j_, "cal"])
                if density <= best_density:
                    continue
                # Check whether adding 5g would breach any ceiling
                delta_fat   = 5 * float(foods.loc[j_, "fat"])
                delta_carbs = 5 * float(foods.loc[j_, "carbs"])
                delta_prot  = 5 * float(foods.loc[j_, "prot"])
                if (_recompute(plan, "fat")   + delta_fat   > user.fat_target   * 1.01 or
                    _recompute(plan, "carbs") + delta_carbs > user.carbs_target * 1.01 or
                    _recompute(plan, "protein") + delta_prot > user.protein_max * 1.01):
                    continue
                best_meal, best_idx, best_density = m, idx, density

        if best_meal is None:
            break   # no item can be increased without breaching a ceiling

        it  = plan[best_meal][best_idx]
        j_  = it["food_idx"]
        ng  = it["grams"] + 5
        it["grams"]     = ng
        it["calories"]  = round(ng * foods.loc[j_, "cal"],       1)
        it["protein"]   = round(ng * foods.loc[j_, "prot"],      1)
        it["fat"]       = round(ng * foods.loc[j_, "fat"],       1)
        it["sat_fat"]   = round(ng * foods.loc[j_, "sat_fat"],   1)
        it["carbs"]     = round(ng * foods.loc[j_, "carbs"],     1)
        it["fiber"]     = round(ng * foods.loc[j_, "fiber"],     1)
        it["net_carbs"] = round(ng * foods.loc[j_, "net_carbs"], 1)
        it["cost"]      = round(ng * foods.loc[j_, "cost"],      2)

    return plan, sol












# ======================================================================
#  SECTION 6 — MULTI-DAY PLAN GENERATOR
# ======================================================================

def day_totals(plan):
    totals = {k: 0.0 for k in
              ["calories", "protein", "fat", "sat_fat",
               "carbs", "fiber", "net_carbs", "cost"]}
    if plan is None:
        return totals
    for meal in MEALS:
        for it in plan.get(meal, []):
            for k in totals:
                totals[k] += it[k]
    return totals


def plan_violates(plan, user):
    # True if the rounded plan misses any macro target.

    if plan is None:
        return True
    t = day_totals(plan)
    tol = ROUNDING_TOL
    violates = (
        t["calories"] < user.cal_target * 0.97
        or t["calories"] > user.cal_target * 1.03
        or t["protein"] < user.protein_target * (1 - tol)
        or t["protein"] > user.protein_max  * (1 + tol)
        or t["fiber"]  < user.fiber_target  * (1 - tol)
        or t["fiber"]  > user.fiber_max     * (1 + tol)
        or t["fat"]    > user.fat_target    * (1 + tol)
        or t["carbs"]  > user.carbs_target  * (1 + tol)
    )
    if user.nc_max_day is not None and t["net_carbs"] > user.nc_max_day * (1 + tol):
        violates = True
    return violates


def build_exclusions(plan, prev_excluded, foods, banned_families=None):
    # Cross-day food exclusion update
    excluded = {k: set(v) for k, v in prev_excluded.items()}
    banned_families = banned_families or set()

    for cat, slot_list in VARY_ACROSS_DAYS.items():
        for meal, s_idx in slot_list:

            # Find the food chosen in this (meal, slot) today
            items = [it for it in plan.get(meal, [])
                     if it["category"] == cat and it["slot"] == s_idx]
            if not items:
                continue
            chosen_idx     = items[0]["food_idx"]
            key            = (meal, s_idx)
            already_banned = excluded.get(key, set())
            fat_thresh     = FAT_FILTER.get(cat, 99)

            # Eligible foods for this slot that are not banned
            alternatives = foods[
                (foods["Category"] == cat) &
                (foods[meal] == 1) &
                (foods["fat"] <= fat_thresh) &
                (~foods.index.isin(already_banned | {chosen_idx})) &
                (~foods["Food_Family"].isin(banned_families))          # FIX
            ]

            # Minimum Alternatives safety check
            if len(alternatives) >= MIN_ALTERNATIVES_TO_BAN:
                excluded.setdefault(key, set()).add(chosen_idx)
            # else: Dont add this food because of too few alternatives remain

    return excluded

def update_family_counts(plan, family_counts):
    counts = dict(family_counts)
    for meal in MEALS:
        for it in plan.get(meal, []):
            fam = it.get("food_family", "other")
            if fam and fam != "other":
                counts[fam] = counts.get(fam, 0) + 1
    return counts


def get_banned_families(family_counts, n_days_remaining):
    # Return food families that have hit their plan-wide limit
    banned = set()
    for family, limit in FAMILY_PLAN_LIMITS.items():
        if family_counts.get(family, 0) >= limit:
            banned.add(family)
    return banned


def _partial_excluded(excluded, keep_last_n=2):
    """
    Keep only the most recently banned food per slot.
    When later days become infeasible because too many foods are banned across
    tracked slots, this intermediate relaxation clears old bans while keeping
    very recent ones
    """

    partial = {}
    for key, banned_set in excluded.items():
        # Keep only the last added items
        if len(banned_set) <= keep_last_n:
            partial[key] = set(banned_set)
        else:
            # Convert to list and keep last keep_last_n
            partial[key] = set(list(banned_set)[-keep_last_n:])
    return partial


def _day_solve_attempts(master_seed, day, excluded, banned_families):
    """
    Relaxation trails to avoid infeasibilty
      1. Normal
      2. No perturbation
      3. Relaxed family: no banned families
      4. Partial exclusions
      5. No exclusions
    slot_off shifts the RNG seed,so trials differ from the primary attempt.
    """
    partial_excl = _partial_excluded(excluded, keep_last_n=2)
    return [
        {"label": None,                 "perturb": True,  "excluded": excluded,      "family_banned": banned_families, "slot_off": 0,    "solve_off": 0},
        {"label": "no perturb",         "perturb": False, "excluded": excluded,      "family_banned": banned_families, "slot_off": 100,  "solve_off": 1},
        {"label": "relaxed family",     "perturb": False, "excluded": excluded,      "family_banned": set(),           "slot_off": 1000, "solve_off": 2},
        {"label": "partial exclusions", "perturb": False, "excluded": partial_excl,  "family_banned": set(),           "slot_off": 1500, "solve_off": 3},  # FIX-5
        {"label": "no exclusions",      "perturb": False, "excluded": {},            "family_banned": set(),           "slot_off": 2000, "solve_off": 4},
    ]


def solve_day_robust(user, foods, day, master_seed, excluded, banned_families):
    # Solving one day, by retrying with relaxations until macros pass or attempts exhausted.
    plan, sol = None, "Infeasible"
    attempts  = _day_solve_attempts(master_seed, day, excluded, banned_families)

    for i, att in enumerate(attempts):
        # Random number for slot activation : master_seed + day + slot_off
        slot_rng = np.random.default_rng(master_seed + day + att["slot_off"])

        triples = build_slot_triples(
            foods,
            excluded=att["excluded"],
            family_banned=att["family_banned"],
            user=user,
            rng=slot_rng,
        )

        plan, sol = solve_one_day(
            user, foods, triples,
            day_label=f"Day{day}" + (f"_{att['solve_off']}" if att["solve_off"] else ""),
            seed=master_seed + day * 17 + att["solve_off"],
            perturb_cost=att["perturb"],
        )
        if plan is not None and not plan_violates(plan, user):
            return plan, sol, att["label"]

        if i < len(attempts) - 1:
            next_label = attempts[i + 1]["label"]
            reason = next_label if plan is None else f"{next_label} (macro miss)"
            print(f"retry ({reason})... ", end="", flush=True)

    return plan, sol, None


def generate_multiday_plan(user, n_days=3, master_seed=None):
    print(user.summary())
    print(f"\n  Generating {n_days}-day plan...")
    print(f"  {'_' * 60}\n")

    foods        = load_foods(allergies=user.allergies)
    excluded     = {}
    family_counts = {}
    results      = []

    if master_seed is None:
        master_seed = int(time.time() * 1000) % 100000
    print(f"  Slot variety seed: {master_seed}  (pass master_seed=N to reproduce)\n")

    for day in range(1, n_days + 1):
        print(f"  Solving Day {day}...", end=" ", flush=True)

        days_remaining  = n_days - day
        banned_families = get_banned_families(family_counts, days_remaining)

        plan, sol, used = solve_day_robust(
            user, foods, day, master_seed, excluded, banned_families,
        )

        if used:
            print(f"[{used}] ", end="", flush=True)
        print(sol)
        results.append((day, plan, sol))

        if plan is not None and not plan_violates(plan, user):
            excluded      = build_exclusions(plan, excluded, foods, banned_families)
            family_counts = update_family_counts(plan, family_counts)

    return results, foods


# ======================================================================
#  SECTION 7 — DISPLAY
# ======================================================================

CAT_LABEL = {
    "V":   "Vegetable",
    "C":   "Carb",
    "P":   "Protein",
    "F":   "Fruit",
    "S":   "Seeds/Nuts",
    "D":   "Dairy / Egg",
    "Fat": "Fat / Oil",
}
W = 70


def print_day(day_num, plan, sol, user):
    if plan is None:
        print(f"\n  DAY {day_num} — No solution [{sol}]\n")
        return

    totals   = {k: 0.0 for k in
                ["calories", "protein", "fat", "sat_fat",
                 "carbs", "fiber", "net_carbs", "cost"]}
    meal_cals = {}

    print(f"\n{'=' * W}")
    print(f"  DAY {day_num}  [{sol}]  —  Target: {user.cal_target:.0f} kcal")
    print(f"{'=' * W}")

    for meal in MEALS:
        items  = plan.get(meal, [])
        m_cal  = sum(it["calories"] for it in items)
        tgt    = user.cal_target * MEAL_CAL_FRAC[meal]
        diff   = m_cal - tgt
        meal_cals[meal] = m_cal

        print(f"\n  {meal.upper():<12}  "
              f"{m_cal:>5.0f} kcal  (target ~{tgt:.0f},  {diff:+.0f})")
        print(f"  {'-' * 66}")
        if not items:
            print("    (no items)")
            continue
        print(f"  {'Food':<35} {'g':>4}  {'kcal':>5}  "
              f"{'P':>5}  {'F':>5}  {'C':>5}  {'Fib':>4}  Type")
        print(f"  {'-' * 66}")
        for it in sorted(items, key=lambda x: -x["calories"]):
            cat = CAT_LABEL.get(it["category"], it["category"])
            fam = f" [{it['food_family']}]" if it["food_family"] not in ("other", "") else ""
            print(f"  {it['food']:<35} {it['grams']:>4}  "
                  f"{it['calories']:>5.0f}  {it['protein']:>5.1f}  "
                  f"{it['fat']:>5.1f}  {it['carbs']:>5.1f}  "
                  f"{it['fiber']:>4.1f}  {cat}{fam}")
        for k in totals:
            totals[k] += sum(it[k] for it in items)

    print(f"\n{'=' * W}")
    print(f"  TOTALS  DAY {day_num}")
    print(f"  {'-' * 66}")
    cal_d = totals["calories"] - user.cal_target
    for label, val, note in [
        ("Calories",  f"{totals['calories']:>7.0f} kcal",
         f"target {user.cal_target:.0f}  ({cal_d:+.0f})"),
        ("Protein",   f"{totals['protein']:>7.1f} g",  f"{user.protein_target} – {user.protein_max} g"),
        ("Fat",       f"{totals['fat']:>7.1f} g",      f"<= {user.fat_target} g"),
        ("Sat. Fat",  f"{totals['sat_fat']:>7.1f} g",
         f"({totals['sat_fat'] * 9 / max(1, totals['calories']) * 100:.1f}% of kcal)"),
        ("Carbs",     f"{totals['carbs']:>7.1f} g",    f"<= {user.carbs_target} g"),
        ("Net carbs", f"{totals['net_carbs']:>7.1f} g",
         f"<= {user.nc_max_day} g" if user.nc_max_day else "no daily cap"),
        ("Fiber",     f"{totals['fiber']:>7.1f} g",    f"{user.fiber_target} – {user.fiber_max} g"),
    ]:
        print(f"  {label:<10}  {val}    {note}")

    print()
    for name, ok, detail in [
        ("Calories",   user.cal_target * 0.97 <= totals["calories"] <= user.cal_target * 1.03,
         f"{totals['calories']:.0f} kcal  ({cal_d:+.0f},  band [{user.cal_target*0.97:.0f}–{user.cal_target*1.03:.0f}])"),
        ("Protein >=", totals["protein"] >= user.protein_target * (1 - ROUNDING_TOL),
         f"{totals['protein']:.1f}/{user.protein_target} g <= {user.protein_max} g" ),
        ("Protein <=", totals["protein"] <= user.protein_max * (1 + ROUNDING_TOL),
         f"{totals['protein']:.1f}/{user.protein_max} g"),
        ("Fat <=",     totals["fat"] <= user.fat_target * (1 + ROUNDING_TOL),
         f"{totals['fat']:.1f}/{user.fat_target} g"),
        ("Carbs <=",   totals["carbs"] <= user.carbs_target * (1 + ROUNDING_TOL),
         f"{totals['carbs']:.1f}/{user.carbs_target} g"),
        ("Fiber >=",   totals["fiber"] >= user.fiber_target * (1 - ROUNDING_TOL),
         f"{totals['fiber']:.1f}/{user.fiber_target} g <= {user.fiber_max} g"),
        ("Fiber <=",   totals["fiber"] <= user.fiber_max * (1 + ROUNDING_TOL),
         f"{totals['fiber']:.1f}/{user.fiber_max} g"),
        *([("Net carbs <=", totals["net_carbs"] <= user.nc_max_day * (1 + ROUNDING_TOL),
            f"{totals['net_carbs']:.1f}/{user.nc_max_day} g")]
          if user.nc_max_day else []),
    ]:
        print(f"  {'OK  ' if ok else 'WARN'}  {name:<12}  {detail}")
    print(f"{'=' * W}\n")


def print_multiday_summary(results, user):
    W2 = 70
    n  = len(results)
    print(f"\n{'#' * W2}")
    print(f"  {n}-DAY PLAN SUMMARY — {user.goal.upper()} — {user.cal_target:.0f} kcal/day")
    print(f"{'#' * W2}")
    for day, plan, sol in results:
        if plan is None:
            continue
        print(f"\n  Day {day}:")
        for meal in MEALS:
            items = ", ".join(it["food"] for it in plan.get(meal, []))
            print(f"    {meal:<12}: {items or '-'}")
    print(f"\n{'#' * W2}\n")


# ======================================================================
#  SECTION 8 — MAIN
# ======================================================================

if __name__ == "__main__":

    user = UserBiometrics(
        age=22,
        gender="male",
        height_cm=172,
        weight_kg=71,
        activity_level="very_Active",
        goal="gain",
        diabetes_status="none",
        hypertension_status="none",   # options: 'none' | 'cardiovascular'
        allergies=[],
    )

    N_DAYS = 6

    results, foods = generate_multiday_plan(user, n_days=N_DAYS)

    for day, plan, sol in results:
        print_day(day, plan, sol, user)

    print_multiday_summary(results, user)