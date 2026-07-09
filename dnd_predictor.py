import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. Load the trained brain
print("Loading True Lethality Engine...")
try:
    model = joblib.load('true_lethality_model.pkl')
except FileNotFoundError:
    print("Error: Could not find 'true_lethality_model.pkl'. Make sure you ran the training script first!")
    exit()

print("\n=== DUNGEON MASTER ENCOUNTER BALANCER ===")
print("Let's test if you are about to TPK your friends.\n")

# 2. Gather Inputs from the DM
try:
    avg_level = float(input("What is the Average Party Level? (e.g., 5.5): "))
    
    print("\n--- Party Composition ---")
    print("Enter the number of players for each class (0 if none):")
    classes = {
        'num_barbarian': int(input("Barbarians: ")),
        'num_bard': int(input("Bards: ")),
        'num_cleric': int(input("Clerics: ")),
        'num_druid': int(input("Druids: ")),
        'num_fighter': int(input("Fighters: ")),
        'num_monk': int(input("Monks: ")),
        'num_paladin': int(input("Paladins: ")),
        'num_ranger': int(input("Rangers: ")),
        'num_rogue': int(input("Rogues: ")),
        'num_sorcerer': int(input("Sorcerers: ")),
        'num_warlock': int(input("Warlocks: ")),
        'num_wizard': int(input("Wizards: "))
    }
    
    print("\n--- Monster Stats ---")
    monster_cr = float(input("Total Encounter CR (e.g., 13.5): "))
    monster_hp = float(input("Total Monster HP: "))
    monster_ac = float(input("Average Monster AC: "))

except ValueError:
    print("Invalid input! Please enter numbers only.")
    exit()

# 3. ON-THE-FLY FEATURE ENGINEERING (The secret sauce)
party_size = sum(classes.values())
if party_size == 0:
    print("You need at least one player!")
    exit()

total_party_level = avg_level * party_size
cr_ratio = monster_cr / total_party_level
hp_per_player = monster_hp / party_size

has_healer = 1 if (classes['num_cleric'] > 0 or classes['num_druid'] > 0 or classes['num_bard'] > 0) else 0
has_tank = 1 if (classes['num_barbarian'] > 0 or classes['num_fighter'] > 0 or classes['num_paladin'] > 0) else 0
has_arcane = 1 if (classes['num_wizard'] > 0 or classes['num_sorcerer'] > 0 or classes['num_warlock'] > 0) else 0
has_martial_dps = 1 if (classes['num_rogue'] > 0 or classes['num_monk'] > 0 or classes['num_ranger'] > 0) else 0

# 4. Package it for the model
input_data = pd.DataFrame([{
    'avg_party_level': avg_level,
    'party_size': party_size,
    'cr_to_party_level': cr_ratio,
    'monster_hp_per_player': hp_per_player,
    'avg_monster_ac': monster_ac,
    'has_healer': has_healer,
    'has_tank': has_tank,
    'has_arcane': has_arcane,
    'has_martial_dps': has_martial_dps
}])

# --- THE OUT-OF-DISTRIBUTION GUARDRAILS ---
warnings_triggered = False

print("\n=========================================")
print("🛡️ DATA VALIDATION CHECKS")
print("=========================================")

if party_size > 7:
    print("⚠️ WARNING: Party size is > 7. This is Out-of-Distribution.")
    print("   The AI was trained on standard party sizes (3-6). Predictions may be inaccurate due to extreme Action Economy.")
    warnings_triggered = True

if hp_per_player < 15 and cr_ratio > 0.2:
    print("⚠️ WARNING: Glass Cannon Detected.")
    print("   The monster has a high CR but incredibly low HP per player. The AI struggles with monsters that die in 1 round.")
    warnings_triggered = True

if warnings_triggered:
    print("-> Please take the AI's prediction with a grain of salt!")
else:
    print("✅ Input data is within normal bounds. AI prediction should be highly accurate.")

# 5. Make the Prediction
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]

win_chance = probabilities[1] * 100
tpk_chance = probabilities[0] * 100

print("\n=========================================")
print("🔮 AI PREDICTION RESULTS")
print("=========================================")
print(f"Engineered CR Ratio: {cr_ratio:.3f}")
print(f"Engineered HP Per Player: {hp_per_player:.1f}")

if prediction == 1:
    print(f"\n✅ OUTCOME: PARTY SURVIVES ({win_chance:.1f}% confidence)")
    if hp_per_player > 40:
        print("⚠️ Warning: High HP per player. The party will win, but it will be a long, exhausting slog.")
else:
    print(f"\n💀 OUTCOME: TOTAL PARTY KILL ({tpk_chance:.1f}% confidence)")
    if has_healer == 0:
        print("💡 AI Advice: The math is deadly. Adding a Healer to the party might flip this result.")
    if cr_ratio > 0.4:
        print("💡 AI Advice: The CR ratio is too high for this party size. Consider nerfing the monster.")
print("=========================================")