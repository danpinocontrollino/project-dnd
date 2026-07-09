import pandas as pd
from sklearn.model_selection import train_test_split
from ctgan import CTGAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("1. Loading Real D&D Data...")
df = pd.read_csv('clean_aggregated_combat_data.csv')

# --- Feature Engineering (Same as before) ---
class_columns = ['num_barbarian', 'num_bard', 'num_cleric', 'num_druid', 'num_fighter', 
                 'num_monk', 'num_paladin', 'num_ranger', 'num_rogue', 'num_sorcerer', 
                 'num_warlock', 'num_wizard']

df['party_size'] = df[class_columns].sum(axis=1).replace(0, 1)
df['avg_party_level'] = df['avg_party_level'].replace(0, 1)
df['true_party_power'] = df['party_size'] * (df['avg_party_level'] ** 1.5)
df['cr_to_party_power'] = df['avg_monster_cr'] / df['true_party_power']
df['monster_hp_per_player'] = df['avg_monster_hp'] / df['party_size']

df['has_healer'] = ((df['num_cleric'] > 0) | (df['num_druid'] > 0) | (df['num_bard'] > 0)).astype(int)
df['has_tank'] = ((df['num_barbarian'] > 0) | (df['num_fighter'] > 0) | (df['num_paladin'] > 0)).astype(int)
df['has_arcane'] = ((df['num_wizard'] > 0) | (df['num_sorcerer'] > 0) | (df['num_warlock'] > 0)).astype(int)
df['has_martial_dps'] = ((df['num_rogue'] > 0) | (df['num_monk'] > 0) | (df['num_ranger'] > 0)).astype(int)

df['target'] = df['final_outcome'].apply(lambda x: 1 if x == 'Party Win' else 0)

features = ['avg_party_level', 'party_size', 'cr_to_party_power', 'monster_hp_per_player', 
            'avg_monster_ac', 'has_healer', 'has_tank', 'has_arcane', 'has_martial_dps']

# Drop NAs
df_model = df[features + ['target']].dropna()

# =========================================================
# THE CRITICAL FIX: Split the data BEFORE generating fakes!
# =========================================================
print("\n2. Splitting into Train and Test Sets...")
X = df_model[features]
y = df_model['target']

# 80% for training, 20% for the locked Test Vault
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# 3. Apply the GAN *ONLY* to the Training Data
# =========================================================
print("\n3. Training GAN only on the Training Losses...")
# Combine X_train and y_train temporarily so the GAN can learn the relationships
train_data = X_train.copy()
train_data['target'] = y_train

# Isolate the losses in the training set
train_losses = train_data[train_data['target'] == 0]

# Train the GAN
discrete_columns = ['party_size', 'has_healer', 'has_tank', 'has_arcane', 'has_martial_dps', 'target']
ctgan = CTGAN(epochs=150, verbose=False) # Reduced epochs for speed, you can bump to 300
ctgan.fit(train_losses, discrete_columns)

# Generate synthetic losses to match the number of wins in the training set
num_wins_in_train = len(train_data[train_data['target'] == 1])
num_losses_in_train = len(train_losses)
needed_synthetic_losses = num_wins_in_train - num_losses_in_train

print(f"Generating {needed_synthetic_losses} synthetic training losses...")
synthetic_losses = ctgan.sample(needed_synthetic_losses)

# Combine the real training data with the synthetic training data
balanced_train_data = pd.concat([train_data, synthetic_losses], ignore_index=True)

X_train_balanced = balanced_train_data[features]
y_train_balanced = balanced_train_data['target']

# =========================================================
# 4. Train and Test the Model
# =========================================================
print("\n4. Training Random Forest on Balanced Data...")
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

print("\n5. Testing Model on 100% PURE REAL DATA...")
y_pred = model.predict(X_test)
print(f"Final True Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))