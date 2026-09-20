-- Nutrition database schema
-- Source of truth: USDA FoodData Central (Foundation Foods + SR Legacy datasets)
-- Design notes:
--   - macros are stored per 100g so portion math is a simple scalar multiply
--   - `food_label_map` decouples our YOLO class names from USDA's food naming,
--     since "fried_rice" (a YOLO class) won't text-match any single USDA entry.
--   - one YOLO label can map to multiple USDA candidates ranked by `priority`,
--     so we can pick a better match later without a schema change (e.g. once
--     we support user cuisine preference or dish-specific overrides).

CREATE TABLE IF NOT EXISTS foods (
    id              BIGSERIAL PRIMARY KEY,
    fdc_id          INTEGER UNIQUE,                -- USDA FoodData Central ID; NULL for user-added foods
    description     TEXT NOT NULL,                 -- e.g. "Rice, fried, no meat" or a user's own food name
    data_type       TEXT NOT NULL,                  -- 'foundation_food' | 'sr_legacy_food' | 'survey_fndds_food' | 'user_added'
    food_category   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Community food library: who added it and whether it's been verified.
    -- USDA-seeded rows are verified=true by definition; user-added rows
    -- start false so you can optionally review before they show up in
    -- search results for other users (see food_library/search.py).
    source          TEXT NOT NULL DEFAULT 'usda',  -- 'usda' | 'user_added'
    added_by_user_id BIGINT,
    verified        BOOLEAN NOT NULL DEFAULT false
);

-- Per-100g macro + key micronutrient values. Kept as a narrow, indexed table
-- (not a generic EAV nutrient table) because the app only ever needs a fixed
-- set of fields per lookup — this keeps the hot-path query a single index scan.
CREATE TABLE IF NOT EXISTS food_nutrients (
    food_id         BIGINT PRIMARY KEY REFERENCES foods(id) ON DELETE CASCADE,
    calories_kcal   NUMERIC(8,2) NOT NULL,
    protein_g       NUMERIC(8,2) NOT NULL DEFAULT 0,
    carbs_g         NUMERIC(8,2) NOT NULL DEFAULT 0,
    fat_g           NUMERIC(8,2) NOT NULL DEFAULT 0,
    fiber_g         NUMERIC(8,2) NOT NULL DEFAULT 0,
    sugar_g         NUMERIC(8,2) NOT NULL DEFAULT 0,
    sodium_mg       NUMERIC(8,2) NOT NULL DEFAULT 0,
    -- typical density, used by the volume estimator to go area -> weight
    -- when no better food-specific heuristic exists. g/cm^3.
    density_g_cm3   NUMERIC(6,3)
);

-- Maps our internal food-recognition class labels (the classes our YOLO
-- model is trained/fine-tuned on) to one or more candidate USDA foods.
-- priority=1 is the default match; lower-priority rows exist so we can
-- disambiguate later (e.g. "curry" -> chicken curry vs veg curry) without
-- a migration.
CREATE TABLE IF NOT EXISTS food_label_map (
    id              BIGSERIAL PRIMARY KEY,
    model_label     TEXT NOT NULL,                 -- e.g. 'fried_rice', 'grilled_chicken_breast'
    food_id         BIGINT NOT NULL REFERENCES foods(id) ON DELETE CASCADE,
    priority        SMALLINT NOT NULL DEFAULT 1,
    -- typical single-serving weight in grams for this label, used as the
    -- portion estimator's prior before reference-object scaling corrects it
    typical_serving_g NUMERIC(8,2),
    UNIQUE (model_label, food_id)
);

CREATE INDEX IF NOT EXISTS idx_food_label_map_label ON food_label_map (model_label, priority);
CREATE INDEX IF NOT EXISTS idx_foods_description_trgm ON foods USING gin (description gin_trgm_ops);
-- requires: CREATE EXTENSION IF NOT EXISTS pg_trgm;  (for fuzzy text fallback matching)
