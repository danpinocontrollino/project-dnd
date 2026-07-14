# 🎓 Guida all'esame — True Lethality Engine

*Spiegazione componente per componente di tutto il progetto. Non è un riassunto:
ogni sezione spiega COSA fa quel pezzo, COME lo fa passo per passo, PERCHÉ è
fatto così e non in un altro modo, con i numeri esatti e il file dove vive.
Ogni numero è riproducibile dal repo e vive in `figures/`. Le domande difficili
con le risposte sono in fondo (§16). Il pitch da 60 secondi è qui sotto.*

---

## 0. Il pitch di 60 secondi

> D&D 5e misura la difficoltà dei mostri con il Challenge Rating, un'euristica
> di design mai validata contro il gioco reale. Noi la sostituiamo con una
> stima **empirica e calibrata di P(il party vince | incontro)**, appresa da
> **34.907 combattimenti reali** (dataset FIREBALL, partite vere su Discord).
> Il modello è un **XGBoost con 45 vincoli monotoni, calibrato con Platt**, su
> 45 feature di matematica del combattimento, validato con **cross-validation
> raggruppata per campagna** (holdout AUC 0.657 [0.638, 0.674], CV 0.601 ±
> 0.040). Dalla probabilità calibrata deriviamo il prodotto: il *True
> Lethality Level*, il livello a cui un party bilanciato di 4 raggiunge il
> tasso di vittoria target, trovato per ricerca binaria. La statistica
> interessante sta in ciò che i dati **non possono** dire: i DM curano gli
> scontri e "perdonano" quelli senza speranza, contaminando esattamente la
> regione che il prodotto interroga. Perciò il modello servito è avvolto in
> tre guardie di dominio esplicite, l'ultima calibrata su **490.640 battaglie
> simulate** con un motore Monte Carlo esterno (Battlecast).

La tesi del progetto in una frase: **un modello può essere giusto sui dati e
sbagliato sulla domanda; il lavoro statistico è accorgersene e correggere con
lo strumento adatto** (vincoli, guardie, simulazione), non con più dati uguali.

---

## 1. Il problema e l'estimand

**Il sistema ufficiale.** Ogni mostro ha un Challenge Rating (CR, da 0 a 30):
"un mostro CR 5 è una sfida equa per quattro eroi di livello 5". Il manuale del
DM (DMG p.82) stima la difficoltà di un incontro così: somma gli XP dei mostri,
applica un moltiplicatore per il loro numero, confronta con soglie per livello.
Il CR è un'intenzione di design: **nessuno l'ha mai validato contro partite
giocate**. Questo è il gap che il progetto attacca.

**La nostra domanda è un estimand, non uno score.** Non chiediamo "che voto
diamo al mostro" ma: η(x) = P(party vince | incontro x), **calibrata**. Poi la
invertiamo: il *True Lethality Level* è il livello del party a cui η supera un
target (default 65%, regolabile dall'utente — "equo" è una **policy**, non una
costante di natura). Due conseguenze progettuali:

1. Tutto a valle consuma **probabilità**, quindi la calibrazione è un
   requisito di prima classe, non un abbellimento. Per questo la metrica
   primaria dichiarata è il **Brier score**, non l'AUC.
2. Ai bordi diamo **verdetti** invece di precisione finta: `trivial` (vince ≥
   target già a livello 1), `beyond_deadly` (sotto target anche a livello 20),
   `ok` (il target viene attraversato dentro [1, 20]).

**Perché "e se combattessero fino alla morte?"** L'app risponde a una domanda
controfattuale/interventista ("questo mostro, contro questo party, chi vince
se nessuno si tira indietro?"), mentre i log rispondono a una domanda
osservazionale ("cosa è successo ai tavoli reali?"). Questa distinzione —
estimand mismatch — è il filo rosso dell'intero progetto (§9, §10).

---

## 2. Glossario D&D minimo (per chi di noi non gioca)

- **Party**: la squadra di eroi, 3–6 personaggi di livello 1–20.
- **HP** (hit points): quanti danni assorbi prima di cadere.
- **AC** (armor class): quanto è difficile colpirti — un attacco colpisce se
  d20 + bonus di attacco ≥ AC.
- **DPR** (damage per round): danno medio inflitto per round, misura del
  danno *sostenuto*.
- **Burst**: la singola azione più pericolosa (es. Power Word Kill del Lich,
  il soffio del drago) — misura del danno *esplosivo*, diversa dal DPR.
- **Save DC**: la difficoltà di resistere agli effetti (incantesimi, veleni).
- **Legendary**: i boss hanno azioni e resistenze extra fuori dal loro turno —
  di fatto più HP e più azioni effettive.
- **Ruoli del party** (come li mappiamo noi): *healer* (Cleric/Druid/Bard),
  *tank* (Barbarian/Fighter/Paladin), *arcane* (Wizard/Sorcerer/Warlock),
  *martial DPS* (Rogue/Monk/Ranger). Nel modello sono 4 flag binari.
- **Action economy**: chi ha più azioni per round vince le guerre di
  logoramento — 4 eroi contro 1 boss = 4 azioni contro ~1–3. È il motivo
  strutturale per cui i boss solitari deludono e gli sciami sorprendono.
- **TPK** (total party kill): il party viene sterminato.
- **DM mercy**: il DM che "perdona" — dadi aggiustati, ritirate concesse,
  rinforzi narrativi. Centrale nel §9.

---

## 3. Il dataset: FIREBALL (e come lo trasformiamo)

**Cos'è.** FIREBALL (Zhu et al., ACL 2023) raccoglie log strutturati di
partite vere giocate su Discord tramite il bot Avrae: ogni riga è uno
snapshot di un turno di combattimento con lo stato del gioco (attori, HP,
livelli, mostri). Non è un sondaggio né una simulazione: è gioco reale, con
tutto il suo rumore.

**La pipeline ETL (`parse_fireball.py`), passo per passo:**

1. **Input**: 1.471 file JSONL (~1.9 GB), uno per "campagna" (il log di un
   server Discord) → 153.829 snapshot di turni.
2. **Ricostruzione degli incontri**: i turni vengono raggruppati in
   combattimenti; 134.475 turni appartengono a combattimenti ben formati.
3. **Etichettatura dall'HP terminale**: un incontro è *Party Win* se lo stato
   finale mostra i mostri a 0 HP / rimossi e il party in piedi. Gli incontri
   **irrisolti** (il log si interrompe, nessuno stato terminale) vengono
   scartati: −85.778 righe. Chi non matcha il bestiario: −9.215.
4. **Aggregazione a livello di incontro**: da righe-turno a UNA riga per
   incontro. Le statistiche dei mostri si aggregano **esattamente come al
   serving** (stessa funzione, `roster_monster_fields`): medie pesate per
   conteggio per le statistiche continue (CR, HP, AC, taglia...), **massimi**
   per i flag binari e per le minacce apicali (max CR, max burst, max save
   DC), **somme** per i totali della corsa al danno (HP totali, DPR totale,
   XP totali).
5. **Sanificazione**: livelli del party clippati a [1, 20], roster a ≤ 30
   mostri (oltre è quasi certamente un errore di log).
6. **Output**: `clean_aggregated_combat_data.csv` — **34.907 incontri**,
   **1.462 campagne**, 29 colonne raw.

**I due numeri da ricordare:**
- **Base rate 83.0%**: il party vince l'83% degli incontri registrati. I DM
  curano gli scontri e i giocatori si ritirano da quelli persi → i mostri
  deboli saturano vicino al soffitto a ogni livello. Conseguenza: verdetti ai
  bordi, slider del target, e attenzione maniacale alla calibrazione.
- **1.462 campagne = 1.462 gruppi di leakage**: incontri della stessa
  campagna condividono party, DM e house rules. È l'unità di scambiabilità
  (§8).

**Domanda tipica — "perché scartate 86k righe invece di imputarle?"**
Perché l'esito mancante non è MCAR: i combattimenti irrisolti sono
sistematicamente diversi (interrotti, fuori dal bot, abbandonati). Imputare
l'esito significherebbe inventare la variabile risposta. Meglio meno righe
oneste che più righe inquinate.

---

## 4. I dati dei mostri e il parsing dell'offesa

**Il problema dell'Attack Potency.** Il bestiario (foglio ufficiale, 802
righe, 762 mostri servibili con statistiche complete) elenca HP, AC, taglia,
attributi — ma **niente su quanto forte colpisce**. Un modello di esito del
combattimento senza il danno del mostro è mezzo cieco. Soluzione in
`monster_offense.py`:

1. **Parsing per-azione degli statblock SRD** (JSON con HTML delle azioni):
   ogni azione viene analizzata separatamente — bonus di attacco, danno medio
   dalle espressioni dei dadi (es. `2d8+4` → 13), save DC. La segmentazione
   per azione è importante: il vecchio regex "pooled" su tutto il blocco
   gonfiava il DPR dei draghi del ~60% sommando pezzi di azioni diverse.
2. **DPR sostenuto vs burst**: il DPR prende l'output ripetibile round per
   round; il **burst** prende la singola azione più letale (soffio, spell
   apicale). Sono feature diverse perché minacciano in modo diverso: il DPR
   vince le guerre di logoramento, il burst elimina un personaggio dal nulla
   (`burst_vs_pc_hp` è top-3 SHAP).
3. **Alternative "or"**: "melee or ranged attack..." → si prende il **max**
   delle alternative, non la somma (bug delle armi versatili: 24 mostri
   raddoppiavano il danno).
4. **Liste di incantesimi**: gli spellcaster spesso non hanno azioni di danno
   nel blocco ma una lista di spell. Una tabella di **30 incantesimi** con
   danno medio da PHB la prezza (il casting sostenuto è cappato alla classe
   fireball ≈ 45). Il Lich esce con DPR 45 e burst 100 (Power Word Kill).
5. **Fallback + clamp di sanità**: 321 mostri hanno offesa parsata dal
   statblock (badge 🎯 nell'app), 441 usano la tabella di design del DMG
   p.274 per CR (badge 📐). Ogni valore parsato è clampato in
   [0.25×, 3×] della banda DMG del suo CR: un errore di regex non può
   produrre un goblin da 500 danni.
6. **Tratti**: un estrattore canonico unico (`extract_official_traits`,
   usato sia dall'ETL che dall'app — una sola fonte di verità) legge
   resistenze fisiche, immunità al crowd-control, resistenza magica, pack
   tactics, spellcasting, rigenerazione, legendary, mobilità.

---

## 5. Il feature engineering (29 colonne raw → 45 feature)

Vive in `DnDFeatureEngineer` (`initial_learn.py`), un transformer sklearn
**dentro la pipeline picklata**: chi chiama `predict` passa righe raw e la
trasformazione è identica in training e serving per costruzione.

**Il modello del party** (il party nei log è descritto solo da livello,
taglia e ruoli — le sue statistiche vanno modellate):
- proficiency = 2 + ⌊(livello−1)/4⌋ (la vera scala 5e)
- bonus di attacco = proficiency + 3 + livello/8
- AC = 14 + livello/4 + 1 se c'è il tank
- DPR per membro = 3 + 1.9·livello (~5 al lv 1, ~12 al 5 con Extra Attack,
  ~24 all'11, ~41 al 20)
- HP per membro = 4.5 + 5.5·livello; il pool totale ×1.25 se c'è un healer

**Le probabilità di colpire** (in entrambe le direzioni):
`hit = clip((21 + bonus_attacco − AC_bersaglio)/20, 0.05, 0.95)` — la
matematica del d20 (serve 21−bonus per colpire AC), con i bordi 5%/95% che
sono le regole reali (1 naturale manca sempre, 20 naturale colpisce sempre).

**Gli HP effettivi del mostro** (mitigazione → HP virtuali):
`eHP = HP_totali × (1 + 0.40·res_fisica + 0.30·legendary + 0.25·rigenerazione)`.
Il DPR effettivo del mostro: `DPR_totale × hit × (1 + 0.4·pack_tactics)`
(pack tactics ≈ vantaggio ≈ +40% di hit rate).

**Le feature della corsa al danno** — il cuore del progetto:
- `rounds_to_kill_party` = HP party / DPR effettivo mostri (clip a 50)
- `rounds_to_kill_monster` = eHP mostri / (DPR party × hit) (clip a 50; la
  versione **non clippata** esiste come colonna extra per la guardia, §10)
- `lethality_log_ratio` = log del rapporto tra i due (clip ±4) — ">0 = il
  party vince la corsa", probabilmente la quantità più informativa di 5e
- `save_dc_pressure` = max save DC − (12 + livello/4), clip [−10, 15]
- `action_economy_ratio` = n mostri / taglia party, clip [0, 10]
- `burst_vs_pc_hp` = max burst / HP di un singolo PC, clip [0, 10] — ">1 =
  un'azione può eliminare un personaggio da piena salute"
- feature di budget XP ufficiali: `xp_budget_ratio` = XP aggiustati / soglia
  deadly del party (il modello può *vedere* cosa direbbe il manuale)

**Il clipping come guardia OOD**: prima di tutto, gli input raw vengono
riportati nelle bande legali di 5e (AC 10–30, HP medi 1–1000 e totali
1–8000, DPR 0.5–350 medio / 0.5–1500 totale, atk 0–20, DC 8–30, burst
0.5–400, attributi 60–250, taglia 1–6). Un mostro homebrew da 10.000 HP entra
nella regione dove gli alberi hanno dati invece di finire su foglie mai
allenate. Essendo dentro il transformer, il clipping è identico ovunque.

**SHAP conferma la teoria del gioco** (`figures/shap_ranking.csv`, top 5 per
|contributo| medio): `avg_party_level` 0.22, `avg_monster_size_num` 0.201,
`burst_vs_pc_hp` 0.187, `num_monsters_total` 0.165, `party_hit_chance` 0.116.
Livello, taglia, burst, action economy, matematica del d20: il modello ha
imparato il gioco, non artefatti.

---

## 6. Il modello: XGBoost con 45 vincoli monotoni

**Perché gradient boosting su alberi**: dati tabulari eterogenei (conteggi,
flag, rapporti), interazioni non lineari attese (il burst conta di più a
livelli bassi), robustezza a scale diverse, e soprattutto il supporto nativo
ai **vincoli di monotonia**.

**I vincoli sono la scelta di modellazione più importante.** Tutte e 45 le
feature hanno un segno imposto (nessuna a 0):
- **10 positivi** (più X ⇒ P(win) mai più bassa): livello e taglia del party,
  i 4 ruoli, potere del party, hit chance del party,
  `rounds_to_kill_party`, `lethality_log_ratio`.
- **35 negativi** (più X ⇒ P(win) mai più alta): tutto ciò che è del mostro —
  numero, CR, HP, AC, DPR, burst, atk, DC, flag di tratto, pressioni
  derivate, `rounds_to_kill_monster`, budget XP.

Doppio ruolo dei vincoli:
1. **Prior di forma**: incodificano conoscenza di dominio certa (più mostri
   non aiutano MAI il party), riducendo la varianza.
2. **Storia OOD**: oltre il supporto dei dati gli alberi vincolati possono
   solo *appiattirsi nella direzione sicura*, mai invertire. Con il clipping
   degli input, è ciò che rende l'extrapolazione degradare con grazia.

**Iperparametri** (Optuna, studio persistito su SQLite — riprendibile, non
rigiocato a mano): 300 alberi, profondità 4, learning rate 0.149,
min_child_weight 7, subsample 0.678, colsample_bytree 0.853, λ=0.105,
α=0.091, γ=0.009. Da notare la profondità 4: la CV raggruppata premia modelli
piccoli — alberi profondi imparano le campagne, non il combattimento.

**Riaddestrare**: `make retrain` (dati → training → 64 test → 13 assiomi).
`initial_learn.py --no-tune` riusa i parametri da `figures/metrics.json`.

---

## 7. La calibrazione (e il fallimento istruttivo dell'isotonica)

**Perché calibrare**: il boosting ottimizza il ranking, ma il prodotto
consuma probabilità (la ricerca binaria cerca "il livello dove P=0.65").

**Come**: `CalibratedClassifierCV` con metodo **sigmoid (Platt)** — una
regressione logistica a 2 parametri sull'output del modello base, appresa su
predizioni out-of-fold. I fold sono **StratifiedGroupKFold(3, seed 42)
raggruppati per campagna**, passati come indici posizionali precalcolati (gli
indici restano validi perché la pipeline preserva l'ordine delle righe): il
modello base e il calibratore non vedono mai la stessa campagna. Senza
questo, il leakage rientrerebbe proprio dalla porta della calibrazione.

**Perché NON isotonica** (domanda quasi certa): l'isotonica è una funzione a
gradini. Sul nostro dominio produceva **solo 14 valori distinti di
probabilità** sull'intera griglia dei livelli 1–20: la ricerca binaria
atterrava sempre sui bordi dei plateau e **1 Lich e 2 Lich ricevevano lo
stesso livello**. In più misurava anche peggio (Brier 0.1397 contro 0.1394
della sigmoide). Lezione da esame: la scelta del calibratore non è solo una
questione di score — la *forma funzionale* deve essere compatibile con l'uso
a valle (qui: invertibilità liscia). La suite comportamentale ha un check
apposito ("curve not a staircase": ≥30 valori distinti sulla griglia).

**Lettura della curva di calibrazione** (`figures/calibration_curve.png`):
aderente alla diagonale dove vive la massa dei dati (0.8–0.9), lieve
sovraconfidenza nei bin bassi (pochi esempi → stima empirica rumorosa).

---

## 8. La valutazione onesta (leakage, split, metriche, bootstrap)

**Il meccanismo del leakage** (da sapere spiegare a memoria): incontri della
stessa campagna condividono party, DM, house rules e stile. Con split casuali
riga per riga, il modello vede in training incontri quasi identici a quelli
di test e "predice" riconoscendo l'impronta della campagna, non valutando il
combattimento. Il numero risultante è gonfiato e non trasferisce a campagne
nuove — che è l'unico caso d'uso reale dell'app.

**Il rimedio**: la campagna è il gruppo, ovunque.
- Split finale: `GroupShuffleSplit` (seed 42) → 27.502 train / 7.106 test,
  campagne disgiunte.
- Model selection: `StratifiedGroupKFold` → CV **0.601 ± 0.040**.
- Perfino i fold della calibrazione (§7) sono raggruppati.

**Le due differenze da non confondere** (Room 13 del deck):
1. CV raggruppata 0.601 vs holdout raggruppato 0.657: **non è leakage** —
   sono entrambi onesti; differiscono perché la CV allena su 4/5 dei dati e
   la varianza tra campagne è alta (±0.040); 0.657 è a ~1.4 σ dalla media.
2. Numeri raggruppati vs numeri con split casuale: **questo è il leakage**.
   Non stampiamo la cifra naive in vetrina (è un numero senza significato per
   il caso d'uso); è riproducibile cambiando splitter. "Reported, not
   shipped."

**Le metriche di holdout** (7.106 incontri di campagne mai viste):
- ROC-AUC **0.657** [0.638, 0.674]
- Brier **0.139** [0.134, 0.145] ← metrica primaria dichiarata
- PR-AUC **0.874** — attenzione: con positivi all'83%, la baseline della
  PR-AUC è ≈ 0.83, quindi è un miglioramento modesto e onesto (trabocchetto
  classico: "perché la PR-AUC è così alta?" → baseline diverse)
- log-loss 0.447, accuracy 0.825 (vicina al base rate 0.830: normale — il
  valore del modello è il ranking e la calibrazione, non superare una
  baseline degenerata sulla classe maggioritaria)

**I CI**: bootstrap percentile, 2.000 ricampionamenti del test set, resample
degeneri (una sola classe) scartati.

**"AUC 0.66 è poco"** — risposta in tre punti: (a) dadi, tattiche e oggetti
magici non registrati sono rumore irriducibile per le feature; (b) la
baseline cieca "vince sempre il party" ha già l'83% di accuratezza, il
margine strutturale è stretto; (c) al prodotto serve ordinare gli incontri e
produrre probabilità calibrate, e 0.66 di AUC con Brier 0.139 lo fa.

**MMD come sanity check dello split** (§13): nessun covariate shift tra
train e holdout → il gap CV/holdout è varianza degli esiti, non shift delle
feature. Lo split design regge.

---

## 9. La scoperta centrale: DM mercy = estimand mismatch

**Il sintomo**: un primo modello valutava **19 Lich (CR 21) battibili da un
party di livello 8**. Assurdo per chiunque conosca il gioco.

**La diagnosi** (il bug era nel mondo, non nel codice): filtrando i log sugli
scontri che la matematica deterministica del danno dichiara senza speranza
(party eliminato in ≤1 round), scopriamo che risultano "vinti"
**l'84.5% delle volte** (659 righe). DM che aggiustano i dadi, ritirate
registrate come vittorie, rinforzi narrativi. Il modello imparava
fedelmente i dati — che però rispondono a "cosa succede ai tavoli", non a
"cosa succede combattendo fino alla morte".

**Perché non si risolve con più dati**: qualsiasi dato raccolto allo stesso
modo porta la stessa contaminazione, perché è generato dallo stesso processo
(la curatela del DM). Serve o (a) un'informazione esterna con l'estimand
giusto — la simulazione — o (b) etichette esplicite di ritirata/fudge (nei
"prossimi passi"). Da qui le guardie.

---

## 10. Le tre guardie (in ordine di applicazione)

Applicate in `predict_win_for_parties`, deterministic e testate:
clipping (dentro il transformer) → modello → dominanza → cap.

**Guardia 1 — prior di forma + clipping** (§5–6): extrapolazione
impossibile da imparare ⇒ vincolata. Il mostro-dio da 10.000 HP/AC 50/DPR 999
passa per il flusso reale dell'app e esce `beyond_deadly`, senza crash.

**Guardia 2 — dominanza del roster**: le medie pesate diluiscono. Lich + 2
Ogre + 6 Goblin ha CR medio 2.94 mentre il Lich da solo è CR 21 → il modello
grezzo valutava lo scontro PIÙ GRANDE più facile (livello 3.25 contro 5.0).
Fix: si valuta il roster completo E ogni sotto-roster omogeneo, e si serve il
**minimo** elementwise di P(win). Assioma: aggiungere mostri non può mai
aiutare il party. È un min su predizioni monotone ⇒ resta monotono.

**Guardia 3 — il cap di deathmatch (v3, ibrido)**. La formula servita:

```
P(win) ≤ min( σ(0.1924·s − 2.0856·ln(k) + 4.1217),  Λ(CR, count, level) )
```

- `s` = round che il party sopravvive; `k` = round che servono per uccidere
  il roster, **non clippato** (la feature del modello si ferma a 50; per la
  guardia un muro da 10.000 HP deve valere ~350 round, non 50 — per questo il
  transformer espone la colonna extra `rounds_to_kill_monster_raw`).
- Il termine di sopravvivenza cattura i wipe rapidi; il termine −ln(k)
  cattura le **sconfitte per logoramento**. I vincoli di segno (A ≥ 0, C ≥ 0)
  rendono il cap dimostrabilmente crescente nel livello e decrescente nel
  numero di mostri ⇒ ricerca binaria e dominanza sopravvivono.
- Fit: logistica **pesata binomialmente** (ogni cella entra con peso pari ai
  trial vinti/persi ⇒ vera massima verosimiglianza su ~408k esiti binari)
  sulle griglie guard+OOD (204 celle), ricostruite con **gli stessi profili
  del bestiario che l'app serve** (v. storia sotto).
- **Λ, il lattice**: due feature di TTK non distinguono un Lich da un sacco
  di punti ferita (le rotazioni di spell e l'AoE non sono prezzate dalla
  matematica del danno). Quindi dove la verità simulata ESISTE, la serviamo
  direttamente: interpolazione trilineare delle P(win) simulate sulla griglia
  dei boss (CR {2,5,10,15,21} × count {1,2,4,8,12,19} × level
  {1,5,9,13,17,20}), resa monotona in fase di build (cummax sul livello,
  cummin su count e CR — solo abbassamenti: il cap non può allentarsi), con
  **astensione sotto CR 2** (i mostri deboli non sono mai stati il bug;
  clampare un goblin alla riga CR 2 strangolerebbe incontri normali). File:
  `battlecast_bridge/guard_lattice.json`. Sulle celle della griglia il numero
  servito È quello simulato (1 Lich, livello 9: 0.282 da entrambe le parti).

**La storia delle versioni** (ottima da raccontare, mostra il metodo):
- **v1** a mano: σ(2.197·s − 4.394), ancore 10/50/90% a 1/2/3 round di
  sopravvivenza. "Una costante a mano è una confessione."
- **v2** primo fit Battlecast: σ(1.6302·s − 3.9771) — meglio (9/33/71%), ma
  ancora **solo sopravvivenza**: contro 1 Lich un party di livello 5
  "sopravvive" 4+ round stimati ⇒ cap 0.95, mentre il simulatore dà 0.000 su
  2.000 deathmatch. In più il fit era calibrato su profili ricostruiti male
  (solo HP/AC con offesa da tabella DMG ⇒ spazio delle feature che il serving
  non produce mai). Risultato assurdo in produzione: 1 Lich "fair" a 3.25.
- **v3** (attuale): race cap + lattice, fit su profili serving-consistenti.

**Il ladder del Lich, prima e dopo, contro il simulatore:**

| Lich | v2 (sbagliato) | v3 (ora) | Battlecast (crossing 65%) |
|---|---|---|---|
| 1 | 3.25 | **11.0** | ≈ 11 |
| 2 | 6.75 | **16.5** | ≈ 16.5 |
| 4 | 12.75 | beyond deadly (7% a lv 20) | 7% a lv 20 |
| 8+ | beyond deadly (9% a lv 20) | beyond deadly (**0.0%**) | 0.0% |

**Perché guardie e non più dati** (carta della slide): ogni guardia blocca un
fallimento che nessun dato raccolto allo stesso modo può correggere —
extrapolazione (là fuori i dati non esistono), diluizione da aggregazione
(artefatto della nostra featurizzazione), estimand mismatch (i log rispondono
a un'altra domanda). Prima la diagnosi, poi lo strumento giusto.

---

## 11. Battlecast: la simulazione come verità esterna

**Cos'è**: battlecast.gg, un simulatore Monte Carlo di combattimenti 5e —
iniziativa, movimento su griglia, slot incantesimo, condizioni, gioco
scriptato "ottimale", **zero misericordia**. Ne abbiamo vendorizzato i tre
asset pubblici (motore invariato) per girare in locale, riproducibile, senza
carico sul servizio: crediti e caveat in `battlecast_bridge/PROVENANCE.md`
(file da NON toccare — è l'attribuzione).

**Le tre griglie disegnate** (304 celle, **490.640 battaglie** in totale;
fino a 2.000 trial per cella — il driver riduce adattivamente per i roster
più grandi; a 2.000 trial l'errore standard della stima è ≤ 0.011):
- **guard** (180 celle): boss {Ankheg CR 2, Air Elemental CR 5, Aboleth CR
  10, Purple Worm CR 15, Lich CR 21} × count {1,2,4,8,12,19} × livello
  {1,5,9,13,17,20} → calibra la guardia (fit + lattice).
- **mercy** (100 celle): 25 mostri SRD, CR 0–10, singolo mostro × livelli
  {3,7,11,15} → quantifica la misericordia (§12). È uno sweep sistematico,
  NON un replay di incontri loggati.
- **OOD** (24 celle): cloni con HP ×{1,5,20} e AC +{0,8} → valida i verdetti
  fuori distribuzione e popola l'angolo "sopravvivi per sempre, non uccidi
  mai" del fit.
- Party fisso: Fighter/Cleric/Wizard/Rogue ai livelli richiesti. Comando:
  `make battlecast` (~15 min). I pareggi/stalli contano come non-vittorie,
  coerente con l'etichetta di training.

**I due caveat dichiarati** (nel report, nelle slide e nei limiti):
1. **Gap di edizione**: Battlecast gioca le regole 2024-SRD (317 statblock),
   i nostri log e il bestiario sono 2014. Esempio concreto: il Lich simulato
   ha 315 HP / AC 20, quello che l'app serve 135 / 17. La calibrazione
   assorbe il gap dichiarandolo.
2. **Gioco scriptato con build ottimizzate** ⇒ i tassi di vittoria simulati
   sono un **upper bound** sulla prestazione dei tavoli reali.

---

## 12. La misericordia quantificata + validazione OOD

Griglia mercy: stessi incontri attraverso due lenti — il nostro modello
(realtà del tavolo) su x... anzi: x = P(win) Battlecast (deathmatch), y =
P(win) modello (tavolo). Risultati (`figures/battlecast_mercy_gap.png`,
`figures/battlecast_summary.json`):
- correlazione **0.842** — le due lenti concordano sull'ordinamento;
- il gap **si incrocia**: sui combattimenti difficili (sim < 0.5, solo 3
  celle) il modello sta ≈ **+0.28 sopra** il simulatore → inflazione da
  misericordia, dichiarata *direzionale* perché 3 celle sono poche; su quelli
  facili (96 celle) sta ≈ **−0.14 sotto** → il soffitto dell'83% (ritirate ed
  etichette rumorose);
- il gioco curato quasi non produce battaglie senza speranza (96/100 celle
  deathmatch-facili): il pericolo al tavolo è il logoramento, non i matchup
  impossibili.

OOD: gli estremi concordano perfettamente (cloni HP×20 vincono ≈ 0.000–0.003
dei deathmatch = il nostro *beyond deadly*), l'ordinamento fine di mezzo è
moderato (Spearman ρ = **0.538**). Sintesi onesta: verdetti OOD validati,
micro-ranking oltre la risoluzione di entrambi i sistemi.

---

## 13. Il confronto con la cassetta degli attrezzi del corso

Tutto in `model_comparison.py` + `gan_ablation.py`, stessa CV raggruppata per
ogni contendente, numeri in `figures/course_benchmark.json`:

| Modello | CV AUC | CV Brier |
|---|---|---|
| Ridge logistic (C=0.03) | **0.655 ± 0.031** | **0.136** |
| RFF kernel logistic | 0.624 ± 0.013 | 0.142 |
| XGBoost vincolato (produzione) | 0.614 ± 0.033 | 0.141 |

**Ridge logistic — il risultato più istruttivo del progetto.** Il modello
LINEARE vince il rischio predittivo osservazionale (le feature di
combat-math portano il segnale; l'albero flessibile overfitta le
idiosincrasie di campagna — "small is the new big"; lo sweep su C era
piatto, C=0.03 non è magico). **Eppure non è in produzione**: promosso
sperimentalmente, valutava 8 Lich "trivial" e il mostro da 10k HP battibile.
I coefficienti confessano il confounding (`figures/logistic_coefficients.png`):
`num_monsters`, `total_dpr`, `burst` **positivi**, e — la pistola fumante —
`avg_party_level` **negativo**: "i party di livello alto perdono di più",
perché ricevono contenuto curato più difficile. Selection confounding da
manuale, leggibile dai coefficienti. L'app fa domande **interventiste**
("stesso mostro, più copie") e solo il modello vincolato sweepa sanamente:
accettiamo ~0.04 di AUC in meno per comprare comportamento decision-grade.
**Predizione ≠ decisione.**

**RFF kernel logistic** (Rahimi–Recht, 500 componenti): approssima una kernel
machine RBF a n=27k dove il kernel esatto sarebbe proibitivo; 0.624 — nessun
vantaggio: il guadagno sta nelle feature, non nella non-linearità del bordo.

**MMD two-sample test** (kernel, non biased): feature di train vs campagne
held-out, kernel RBF con median heuristic (bandwidth 6.02), 1.500 punti per
lato, 200 permutazioni. MMD² osservato = 0.00031 < soglia 95% del null =
0.00046, **p = 0.12** → nessun covariate shift rilevabile. Il gap CV/holdout
è varianza degli esiti a livello campagna, non shift delle feature: il
design dello split regge.

**Gaussian Process** sul task ausiliario "predici il CR ufficiale dalle
statistiche" (n = 797: 637 train / 160 test): kernel ARD RBF (12 length
scale apprese, una per feature) + WhiteKernel (rumore 0.043), fit per
marginal likelihood. MAE **0.86** contro 0.94 di XGBoost, R² 0.954, e gli
intervalli predittivi al 95% hanno copertura empirica **0.95** — la
calibrazione dell'incertezza che il boosting non dà. (Candidato upgrade per
il CR predictor dell'app, con display dell'incertezza.)

**CTGAN ablation — perché bilanciare fa male** (`gan_ablation.py`): CTGAN
addestrato SOLO su sconfitte delle campagne di training (leakage-guarded),
23.091 righe sintetiche per bilanciare l'83/17. Stesso holdout: AUC scende a
0.638, il Brier ESPLODE a 0.186, la P(win) media predetta crolla a 0.61 in un
mondo all'83%. Spiegazione da esame: il log-loss è una **proper scoring
rule** — il suo minimo è la vera probabilità condizionata, quindi il modello
impara correttamente anche da classi sbilanciate. Bilanciare insegna un base
rate falso e distrugge la calibrazione per riparare un problema che le
proper loss non hanno mai avuto.

---

## 14. Dal modello al prodotto

**L'appraisal** (`lethality_appraisal`): ricerca binaria su [1, 20], 18
iterazioni (risoluzione ~0.0001 di livello), party bilanciato di 4, target
0.65 di default; il risultato è arrotondato al **quarto di livello** (la
risoluzione fine sarebbe finta precisione). Prima della ricerca si valutano i
bordi: P(lv 1) ≥ target ⇒ `trivial`; P(lv 20) < target ⇒ `beyond_deadly`.
Insieme al livello si riporta **p_at_level** (la P(win) esatta al livello
appraisato): due incontri possono condividere il livello e differire in
rischio — la probabilità è il prodotto, il livello è il riassunto.

**La matematica del manuale** (`official_encounter_estimate`, DMG p.82 vera):
1. somma gli XP dei mostri (CR → XP dalla tabella ufficiale, 34 righe);
2. moltiplicatore per numero: ×1 (1), ×1.5 (2), ×2 (3–6), ×2.5 (7–10),
   ×3 (11–14), ×4 (15+), con lo scalino p.83 per la taglia del party (<3
   giocatori: un gradino su; ≥6: un gradino giù);
3. XP aggiustati → CR equivalente con `xp_to_cr` (interpolazione lineare tra
   le righe della tabella, clampata a [0, 30]; round-trip esatto sui valori
   di tabella, così un singolo mostro mostra sempre il suo CR stampato).
Esempio che era un bug: 6 × CR 1 = 1.200 XP × 2 = 2.400 XP ⇒ ~CR 6, non
"CR 1". Coperto da 9 test di regressione incluso l'esempio svolto del DMG.
L'app mostra SEMPRE le due colonne fianco a fianco: "by the book" (con i
passaggi) vs True Lethality Level.

**L'app Streamlit** (dnd-appraising.streamlit.app, deploy automatico da
`main`): 3 tab — 📖 bestiario ufficiale (762 mostri, badge di provenienza
dell'offesa 🎯 parsata / 📐 tabella DMG), 🛠️ homebrew appraiser (statistiche
raw → CR predetto "cosa stamperebbe WotC" + livello di letalità), 🧟
encounter builder (roster misti, aggregati ESATTAMENTE come in training,
protetti dalla dominanza). Curve di P(win) per composizione del party,
heatmap taglia×livello, slider del target. Gemello da terminale:
`fair_fight_finder.py`.

---

## 15. L'ingegneria (riproducibilità e test)

- **64 test pytest** (`make test`): `test_monster_offense` (regex, spell
  table, alternative "or", clamp), `test_feature_engineering` (formule,
  clip, ordine colonne), `test_book_math` (9 test: round-trip xp/cr, scala
  dei moltiplicatori, esempio del DMG, monotonia), `test_survival_guard`
  (12 test: ancore del lattice = valori simulati esatti, costanti fittate,
  muro da 10k HP, astensione < CR 2, monotonia in livello e count).
- **13 assiomi comportamentali** (`make verify`, `behavior_suite.py`): scala
  con il numero di lich monotona; dominanza; curva monotona nel livello; "no
  staircase" (≥30 valori distinti); 11× e 19× Lich beyond_deadly; curva
  GUARDATA ancora monotona; ordinamento dei tier (goblin > ogre > aboleth >
  tarrasque a lv 1); goblin trivial; tarrasque beyond_deadly; mostro-dio da
  10k HP attraverso il VERO flusso app (CR predictor → appraisal) beyond
  deadly. Gate obbligatorio per ogni cambio di modello.
- **Makefile**: `data`, `train`, `tune`, `test`, `verify`, `retrain` (tutta
  la pipeline), `benchmark`, `battlecast`, `slides`, `report`,
  `present-data`, `present`, `app`, `cli`, `clean`.
- **Tracciabilità**: `figures/experiments.jsonl` append-only (ogni run di
  training aggiunge una riga: parametri, metriche, timestamp); studio Optuna
  persistito su SQLite; `figures/metrics.json` con CI; ambiente pinnato in
  requirements.txt (pin VERI: pandas 3.0.3, numpy 2.5.1, xgboost 3.3.0,
  sklearn 1.9.0, streamlit 1.59.1).
- **analyze.py si difende da solo**: se le costanti della guardia in
  produzione derivano dal suo fit, stampa un warning ("update _GUARD_*").

---

## 16. Limiti dichiarati + le domande difficili (con risposte)

**"Perché non una rete neurale / deep learning?"**
34.907 righe tabulari con leakage di gruppo: il regime dove gli alberi
vincolati e i modelli lineari dominano. Una rete non offrirebbe né i vincoli
di monotonia nativi né interpretabilità, e la CV raggruppata (varianza ±0.04)
non avrebbe il potere di distinguerla. Il collo di bottiglia sono le
etichette rumorose, non la capacità del modello.

**"Perché non simulare TUTTO il dataset con Battlecast e mollare FIREBALL?"**
(a) Impareremmo solo Battlecast, coi suoi bias (regole 2024, party fisso,
gioco scriptato, niente terreno/lair) e senza segnale indipendente per capire
DOVE sbaglia; (b) l'input space dell'app include homebrew arbitrario che il
simulatore non rappresenta; (c) il valore statistico del progetto (leakage,
confounding, calibrazione su base rate estremo, estimand mismatch) nasce
proprio dai dati osservazionali; (d) se vuoi la risposta di Battlecast,
esegui Battlecast. L'architettura finale è una **triangolazione**: dati reali
dove i tavoli sono affidabili, simulatore dove i dati reali mentono in modo
documentato. (Esperimento futuro citabile: modello gemello solo-sim e misura
del sim-to-real gap su FIREBALL.)

**"Il race cap da solo è ancora morbido nella zona di crossing."**
Vero, e lo diciamo: celle con gli stessi (s, k) hanno esiti simulati opposti
— l'errore è nelle feature (l'AoE non è prezzata), non nel fit. È esattamente
il motivo per cui esiste il lattice: dove la verità simulata c'è, il min la
serve tal quale.

**"Ma allora il vostro modello sui boss è solo il simulatore?"**
Sui boss della griglia, il TETTO è il simulatore; il modello resta il
predittore ovunque il cap non morde (cioè quasi ovunque: il cap è inerte
sugli incontri normali — goblin, sciami, boss sotto-livello). Il prodotto
dichiara l'estimand "fino alla morte", e su quella domanda il simulatore è
l'oracolo migliore che abbiamo.

**"Perché l'accuratezza (0.825) è sotto il base rate (0.830)?"**
Perché non ottimizziamo la 0/1 loss: soglia a 0.5 su probabilità calibrate in
un mondo all'83% classifica quasi tutto positivo. Il prodotto non usa mai la
soglia: usa le probabilità. Brier e log-loss sono le metriche giuste.

**"Quanto è alto il numero naive (split casuale)?"**
Non lo pubblichiamo (numero senza significato per il caso d'uso); è
riproducibile cambiando lo splitter, e viene sensibilmente più alto. Non
citare cifre a memoria che non sono scritte nel repo.

**"Il party model (3+1.9·lvl ecc.) è rozzo."**
Sì, deliberatamente: è un prior di scala lineare nella sola variabile
osservata (il livello), usato in modo identico in training e serving. Ogni
raffinamento (build, oggetti) richiederebbe dati che i log non hanno. Gli
errori sistematici del party model vengono assorbiti dal modello a valle
(che vede le feature, non la "verità").

**"+0.28 di mercy su 3 celle è statistica?"**
No, ed è per questo che la chiamiamo *direzionale* in ogni artefatto. Il
gioco curato quasi non produce scontri senza speranza — è un fatto sul
processo generativo, e la coda del report propone il fix: etichette esplicite
di ritirata/fudge, che trasformerebbero la misericordia da contaminazione a
variabile di censura.

**"Perché il target 65% e il quarto di livello?"**
Policy, non natura: 65% è il default "sfida equa" e c'è uno slider. Il quarto
di livello è la risoluzione oltre la quale la precisione sarebbe finta
(l'incertezza del modello è ben più larga di 0.25 livelli).

**"Draw nel simulatore?"** Contano come non-vittorie, coerente con
l'etichetta di training (solo "Party Win" è positivo).

**"Perché il GP non è in produzione se batte XGBoost sul CR?"**
Task ausiliario (797 mostri): il GP vince e dà intervalli con copertura
esatta — è il candidato upgrade dichiarato per il CR predictor. Non è il
modello principale perché n=27k con feature engineering pesante è terreno da
boosting, e il GP esatto scala O(n³).

**"2024 vs 2014?"** Il simulatore gioca statblock 2024, i log e il bestiario
sono 2014 (Lich: 315/20 vs 135/17). La calibrazione della guardia assorbe il
gap e lo dichiara; i tassi simulati restano upper bound (build ottimizzate).

**Limiti che ammettiamo per primi** (Room 24 del deck): rumore irriducibile
(AUC 0.657 in un mondo all'83%); estimand mismatch gestito con un cap
esplicito e testato, non imparato in silenzio; gap di edizione del
simulatore; mercy direzionale su poche celle; il micro-ranking OOD oltre la
risoluzione di entrambi i sistemi.

---

## 17. Chi presenta cosa + comandi da sapere

**I 5 atti del deck interattivo** (`presentation/slides_interactive.html`,
`make present` → localhost:8765):
- **Atto I — The broken rulebook** (Daniele): il gioco, il CR, la matematica
  del DMG dal vivo (calcolatrice), perché è rotta.
- **Atto II — The data** (Francesca): FIREBALL, l'ETL, l'Attack Potency, le
  feature + SHAP.
- **Atto III — Model & evaluation** (Stefano): XGBoost vincolato,
  calibrazione, valutazione onesta, il ridge che vince e perde, il toolbox
  del corso.
- **Atto IV — Guards & simulation** (Antonietta): il quiz dei 19 Lich, le
  tre guardie, il race cap (widget a due slider), il Lich Lab, la mercy.
- **Atto V — The reckoning** (Daniele): il prodotto live, i limiti, i cinque
  takeaway.

**Comandi da sapere a memoria all'esame**:
`make retrain` (pipeline completa) · `make verify` (13 assiomi) · `make test`
(64 test) · `make benchmark` (toolbox) · `make battlecast` (griglie + fit
guardia) · `make app` / app live su **dnd-appraising.streamlit.app** (HEAD di
main) · `make report` / `make slides` (PDF).

**I numeri da sapere a memoria**: 34.907 incontri / 1.462 campagne / base
rate 83.0% / 45 feature e 45 vincoli / AUC 0.657 [0.638, 0.674] / Brier
0.139 / CV 0.601 ± 0.040 / 84.5% mercy su 659 righe / 490.640 battaglie /
ladder 1→11, 2→16.5, 3+→beyond deadly / ridge 0.655 ma respinta / GAN
0.186 di Brier / 64 test + 13 assiomi.
