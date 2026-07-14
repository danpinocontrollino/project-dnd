# 🐉 True Lethality Engine — Guida Completa (Italiano)

> **Prevedere la vera pericolosità dei combattimenti di D&D 5ª Edizione con il Machine Learning**
> Progetto finale — Statistical Machine Learning, Sapienza Università di Roma

Questa guida spiega **tutto**: il gioco e le sue regole, ogni sezione del sito, ogni numero, flag e definizione che compare a schermo, il modello statistico e l'architettura del codice.

App online: **https://dnd-appraising.streamlit.app**

---

## Indice

1. [Il problema: cos'è il Grado di Sfida e perché è rotto](#1-il-problema)
2. [Glossario di gioco: tutte le statistiche e i tratti](#2-glossario-di-gioco)
3. [Il sito, sezione per sezione](#3-il-sito-sezione-per-sezione)
4. [I numeri del modello: cosa significano davvero](#4-i-numeri-del-modello)
5. [Le tre "guardie" del motore](#5-le-tre-guardie-del-motore)
6. [Architettura del codice, file per file](#6-architettura-del-codice)
7. [Come eseguire tutto](#7-come-eseguire-tutto)
8. [Limiti e onestà scientifica](#8-limiti-e-onestà-scientifica)

---

## 1. Il problema

In **Dungeons & Dragons 5ª Edizione** un gruppo di giocatori (il *party*) affronta mostri controllati dal Dungeon Master (DM). Per aiutare il DM a bilanciare gli scontri, il manuale ufficiale assegna a ogni mostro un **Grado di Sfida** (*Challenge Rating*, **CR**): in teoria, un mostro di CR 5 dovrebbe essere una sfida equa per quattro personaggi di livello 5.

In pratica il CR è una **intenzione di design**, non una misura empirica: ignora la composizione del party, l'economia delle azioni, le sinergie tra tratti dei mostri, e scala male ai livelli alti. Questo progetto lo sostituisce con una misura **empirica**: un modello XGBoost calibrato, addestrato su **34.907 combattimenti reali** estratti dal dataset FIREBALL (log di partite vere giocate su Discord tramite il bot Avrae), che stima la **probabilità reale di vittoria** del party.

Da quella probabilità deriviamo il **True Lethality Level**: il livello del party al quale un gruppo bilanciato di 4 personaggi ha esattamente la probabilità di vittoria scelta come bersaglio (default 65%).

---

## 2. Glossario di gioco

### Statistiche di base del mostro

| Statistica | Nome nel sito | Significato |
|---|---|---|
| **CR** | Challenge Rating / Grado di Sfida | La difficoltà ufficiale assegnata da Wizards of the Coast. Va da 0 a 30, con frazioni (1/8, 1/4, 1/2) per i mostri deboli. |
| **HP** | Hit Points / Punti Ferita | Quanto danno il mostro può assorbire prima di morire. |
| **AC** | Armor Class / Classe Armatura | Quanto è difficile colpirlo: un attacco va a segno se `d20 + bonus di attacco ≥ AC`. |
| **DPR** | Damage Per Round / Danno per Round | Il danno medio che il mostro infligge in un round usando la sua routine d'attacco normale (multiattacco incluso). Estratto **dal testo reale degli statblock SRD** ("+9 to hit", "12 (2d6+5)", "makes three attacks…"); per i mostri non-SRD è imputato dalla tabella di design del DMG p.274. |
| **Burst** | Burst damage / Danno nova | Il danno della **singola azione più spaventosa** del mostro: soffio di drago, incantesimo più forte (il Power Word Kill del Lich vale 100), ecc. Diverso dal DPR, che è il danno *sostenuto*. |
| **+X to hit** | Attack bonus / Bonus di attacco | Il bonus che il mostro somma al d20 per colpire. Confrontato con la AC stimata del party determina la probabilità di colpire. |
| **DC** | Save DC / Classe Difficoltà dei tiri salvezza | Quando il mostro impone un tiro salvezza (veleno, paralisi, soffio…), il personaggio deve superare questa soglia. Più è alta, più è pericoloso. |
| **Ability score sum** | Somma caratteristiche | La somma dei sei punteggi di caratteristica (FOR+DES+COS+INT+SAG+CAR). Un goblin è ~60, un drago adulto ~130, una divinità ~180. Misura il "budget" complessivo del mostro. |
| **Size** | Taglia | Tiny → Small → Medium → Large → Huge → Gargantuan. Codificata 1–6. |
| **XP** | Experience Points / Punti Esperienza | Il "valore" del mostro secondo la tabella ufficiale CR→XP (DMG p.274): CR 1 = 200 XP, CR 5 = 1.800, CR 21 = 33.000, ecc. |

### I tratti (le checkbox / flag del sito)

| Flag | Significato in gioco | Perché conta per il modello |
|---|---|---|
| **Legendary** | Il mostro ha Azioni Leggendarie (agisce anche fuori dal suo turno) e/o Resistenze Leggendarie (annulla i tiri salvezza falliti). | Vince la guerra di logoramento: senza un guaritore il party crolla. |
| **High mobility (fly/swim)** | Vola o nuota. | Un party solo da mischia non riesce nemmeno a toccarlo. |
| **Regeneration** | Recupera punti ferita ogni round (es. Troll). | Allunga lo scontro; punisce i party con poco danno. |
| **Nonmagical physical resistance** | Dimezza il danno di armi non magiche. | Dimezza di fatto il contributo dei personaggi marziali. |
| **Magic resistance** | Vantaggio ai tiri salvezza contro incantesimi. | Gli incantatori faticano a far "attaccare" i propri incantesimi. |
| **Immune to hard CC** | Immune a stordito/paralizzato/affascinato/spaventato. | Le tattiche di controllo (la via più facile per vincere) non funzionano. |
| **Spellcaster** | Lancia incantesimi. | Danno ad area, controllo, burst imprevedibile. |
| **Pack tactics** | Ha vantaggio agli attacchi se un alleato è adiacente al bersaglio. | In branco colpisce quasi sempre: il pericolo cresce più che linearmente col numero. |

### Il party

| Termine | Definizione |
|---|---|
| **Party size** | Numero di personaggi (il sito simula gruppi da 3 a 6). |
| **Average party level** | Livello medio dei personaggi (1–20). |
| **Healer** | Chierico, Druido o Bardo: cura e rimuove condizioni. |
| **Tank** | Barbaro, Guerriero o Paladino: prima linea con tanta AC/HP. |
| **Arcane** | Mago, Stregone o Warlock: danno magico a distanza e controllo. |
| **Martial DPS** | Ladro, Monaco o Ranger: danno d'arma sostenuto, spesso a distanza. |

### Le 5 composizioni simulate

| Composizione | Ruoli presenti | Stile |
|---|---|---|
| **Balanced** | Healer + Tank + Arcane + Martial DPS | Il party "da manuale": ha una risposta a tutto. È il riferimento per il True Lethality Level. |
| **Glass Cannons** | Arcane + Martial DPS | Tutto attacco, niente difese: uccide in fretta ma crolla se il mostro sopravvive. |
| **The Wall** | Healer + Tank | Quasi immortale ma lentissimo a chiudere: soffre i volanti e i rigeneranti. |
| **Melee Rush** | Tank + Martial DPS | Picchiatori da mischia: ottimi contro i bruti a terra, impotenti contro chi vola. |
| **Full Caster** | Healer + Arcane | Controllo e distanza, ma la retrovia è fragile: i branchi in mischia la raggiungono. |

### La matematica ufficiale del manuale (DMG p.82)

Il DMG valuta un incontro così:

1. **Somma gli XP** di tutti i mostri.
2. **Applica il moltiplicatore per numero di mostri** (l'economia delle azioni): ×1 per 1 mostro, ×1,5 per 2, ×2 per 3–6, ×2,5 per 7–10, ×3 per 11–14, ×4 per 15+. Con meno di 3 giocatori si sale di uno scalino, con 6+ si scende.
3. **Confronta gli XP modificati** con le soglie per livello del personaggio (Facile / Medio / Difficile / Letale — la tabella "Soglie di PE" del DMG). Nel sito aggiungiamo **☠️ Super-Deadly** per gli incontri oltre il doppio della soglia Letale.

Il sito esegue questo calcolo **per davvero**: 6 mostri di CR 1 = 1.200 XP × ×2 = 2.400 XP modificati ≈ **un singolo mostro di CR 6** — non "CR 1".

---

## 3. Il sito, sezione per sezione

### 3.1 Barra laterale (sidebar)

- **🐉 True Lethality Engine** — presentazione: il motore è addestrato su 34.907 incontri reali e usa DPR/bonus di attacco/DC estratti da statblock veri.
- **Model card (held-out campaigns)** — le metriche di qualità del modello misurate su **campagne mai viste in addestramento** (vedi §4):
  - **ROC-AUC** (~0,65): capacità di distinguere vittorie da sconfitte;
  - **Brier** (~0,14): errore quadratico delle probabilità (più basso = probabilità più affidabili);
  - la nota sotto indica **CV raggruppata**, **calibrazione Platt** e **vincoli monotoni**.
- **🎯 Target win rate** (slider 0,50–0,90, default 0,65) — definisce cosa intendi per "scontro equo": la probabilità di vittoria che il True Lethality Level deve raggiungere. Nota: nei log reali i party vincono l'83% degli scontri, quindi 0,85 ≈ "incontro tipico curato dal DM" e 0,55 ≈ "vera moneta lanciata".

### 3.2 «📚 What do the party compositions mean?»

Il pannello espandibile sopra le tab: due tabelle che spiegano i **quattro ruoli** (quali classi vi appartengono e perché contano tatticamente) e le **cinque composizioni** con i ruoli presenti (✅) e lo stile di gioco. Sono le stesse definizioni del glossario qui sopra, §2.

### 3.3 Tab «📖 Official Monster»

1. **Search the official bestiary** — menu a tendina con ~760 mostri ufficiali.
2. **Banner verde** — le statistiche del mostro scelto: `Lich — CR 21 · 135 HP · AC 17 · DPR 45 · burst 100 · +12 to hit · DC 20` più un badge:
   - **🎯 real statblock** = le statistiche offensive sono state *lette dal testo reale* dello statblock SRD;
   - **📐 DMG design table** = il mostro non è nell'SRD, quindi DPR/attacco/DC sono i valori attesi per il suo CR secondo la tabella di design del DMG p.274.
3. **Number of monsters (action economy)** — quante copie del mostro affronti (1–30).
4. **Calculate True Lethality** — lancia l'appraisal (vedi §3.6).

### 3.4 Tab «🛠️ Homebrew Appraiser»

Per mostri inventati da te. Campi:

- **Hit points, Armor class, Ability score sum, Size** — le statistiche difensive di base;
- **Offense (vuoto = stima automatica)** — DPR, bonus di attacco, DC e Burst: se li lasci vuoti vengono stimati dalla tabella DMG in base al CR predetto;
- **Le 8 checkbox dei tratti** — vedi glossario §2;
- **Number of monsters**.

Alla pressione del bottone il sito mostra prima **🤖 Predicted WotC rating**: un secondo modello (regressore XGBoost addestrato sui ~800 mostri ufficiali, MAE ≈ 0,9 CR) stima *che CR gli darebbe Wizards of the Coast*. Quel CR fa da base per il calcolo XP e per il confronto col motore.

### 3.5 Tab «🧟 Encounter Builder»

Per incontri **misti**: mostri diversi, ufficiali e homebrew insieme.

- **Add official monster** — scegli dal bestiario + quantità;
- **Add homebrew monster** — modulo compatto con le stesse voci della tab 2;
- **Current encounter** — la lista del roster con le statistiche di ciascuno e il cestino 🗑️ per rimuovere;
- **Totals** — riepilogo aggregato: numero mostri, HP totali, DPR totale, CR massimo ("apex"), XP totali;
- **Calculate True Lethality / Clear encounter**.

L'aggregazione replica **esattamente** quella usata in addestramento: medie pesate per le statistiche continue, massimo per i flag e per le minacce apicali, somme per HP/DPR/XP.

### 3.6 La schermata dei risultati (comune a tutte le tab)

**Le tre metriche in alto:**

1. **📖 Official Monster Manual** (o "Predicted WotC rating" / etichetta del roster) — la valutazione *del manuale*. Con 1 mostro è il suo CR stampato; con più mostri è il **CR-equivalente secondo il DMG p.82** (somma XP × moltiplicatore → CR del mostro singolo equivalente), col tag "N monsters, DMG-adjusted" e, sotto, la riga *"📖 Book math"* che mostra l'aritmetica completa.
2. **⚡ True Lethality Level** — la risposta del modello: il livello del party (gruppo bilanciato di 4) al quale la probabilità di vittoria raggiunge il target. Sotto: la **percentuale esatta di vittoria a quel livello** (due incontri possono condividere il livello ma avere rischio diverso). Valori speciali:
   - **≤ 1** = *banale*: persino un party di livello 1 vince oltre il target;
   - **> 20** = *oltre il letale*: nemmeno un party di livello 20 raggiunge il target.
3. **P(win) @ level 1 → 20** — la probabilità di vittoria agli estremi della scala: fotografa quanto lo scontro "scala" col livello.

**I messaggi di verdetto:**

- 🕊️ *Trivial* — anche a livello 1 si vince quasi sempre (i mostri deboli saturano presto: nei log reali i party vincono comunque);
- ☠️ *Beyond deadly — TPK machine* — nemmeno a livello 20 si raggiunge il target (TPK = *Total Party Kill*, sterminio del gruppo);
- 🚨 *Major discrepancy* — il motore e il manuale (calcolato onestamente!) divergono di ≥2 livelli.

**Il grafico «Win probability vs. party level»** — cinque curve (una per composizione, colori fissi) della probabilità di vittoria da livello 1 a 20; la banda grigia è la zona "equa" (target ±10%), la linea tratteggiata orizzontale è il target, la linea verticale punteggiata marca il True Lethality Level. Passandoci sopra col mouse leggi i valori esatti.

**La heatmap «Balanced-party win probability by size and level»** — probabilità di vittoria per ogni combinazione taglia-party (3–6 giocatori) × livello (1–20), blu chiaro → blu scuro. Serve a rispondere a "e se fossimo in 5?".

**«⚔️ Fairest matchups»** — la tabella dei party più vicini al target tra i 400 simulati (4 taglie × 20 livelli × 5 composizioni). Colonne:
- *Party level*, *Size*, *Composition* — chi sono;
- *Win %* — la probabilità prevista;
- *DMG tier* — come il manuale classificherebbe quell'accoppiamento (Easy/Medium/Hard/Deadly/☠️ Super-Deadly), usando gli XP modificati.

**I threat flags (le righe grigie sotto la tabella):** avvisi tattici attivati dai tratti del mostro:
- 👑 *Legendary* — porta un guaritore per la guerra di logoramento;
- 🪽 *Fly/swim* — i party solo-mischia soffriranno;
- 🛡️ *Magic resistance* — gli incantatori perdono valore;
- ⚔️ *Physical resistance* — il danno marziale è dimezzato;
- 🐺 *Pack tactics × più mostri* — vantaggio ovunque;
- 💥 *Nova threat* — la sua azione singola più forte infligge ~N danni (può abbattere un personaggio in un colpo).

---

## 4. I numeri del modello

> **Metrica primaria: ROC-AUC in cross-validation raggruppata per campagna** — guida la scelta degli iperparametri e il confronto tra modelli. **Guardrail di calibrazione: punteggio di Brier** — il sito consuma probabilità grezze, quindi nessun modello può barattare calibrazione per discriminazione. Le metriche di holdout riportano **intervalli di confidenza bootstrap al 95%** (2.000 ricampionamenti, in `figures/metrics.json`) e ogni addestramento viene registrato in `figures/experiments.jsonl`.

- **P(win)** — la probabilità che il party vinca lo scontro, stimata dal modello e **calibrata**: quando il sito dice 70%, su cento scontri simili nei dati reali circa 70 finiscono in vittoria. La calibrazione usa il metodo di **Platt (sigmoide)**; l'alternativa isotonica è stata provata e scartata perché produceva probabilità "a gradini" che rendevano identici incontri diversi.
- **ROC-AUC ≈ 0,65** — la probabilità che, presi a caso uno scontro vinto e uno perso, il modello assegni P(win) più alta a quello vinto. 0,5 = moneta, 1 = perfetto. Il valore è **onesto**: misurato con **validazione raggruppata per campagna** (`StratifiedGroupKFold`): gli scontri della stessa campagna Discord condividono party, DM e house rules, quindi finire nello stesso split di train e test gonfierebbe i numeri. Con split casuali si ottengono metriche più alte — ma è *leakage*, non bravura.
- **Brier ≈ 0,14** — errore quadratico medio delle probabilità. Più informativo dell'accuratezza quando le classi sono sbilanciate (l'83% degli scontri reali è una vittoria).
- **True Lethality Level** — trovato con **ricerca binaria** sul livello del party (1–20) contro un party bilanciato di 4, fino a incrociare il target.
- **Il modello** — XGBoost (300 alberi, profondità 4, iperparametri scelti da Optuna sotto CV raggruppata) dentro una pipeline che include tutta la feature engineering: rapporto CR/potenza del party, probabilità di colpire da entrambi i lati, corsa al danno (*time-to-kill*), pressione dei tiri salvezza, budget XP col moltiplicatore, interazioni tratto-composizione (es. *magic resistance × party arcano*). Nota istruttiva dal benchmark del corso: una regressione logistica ridge **batte** XGBoost in AUC osservazionale (0,657 vs 0,614) ma è stata **scartata** perché senza vincoli monotoni rispondeva in modo assurdo alle domande controfattuali dell'app (8 Lich = banale). Predire bene non basta: bisogna *decidere* bene.

---

## 5. Le tre "guardie" del motore

Il modello impara dai dati; queste tre guardie gli impediscono di dire sciocchezze dove i dati non bastano.

1. **Vincoli monotoni + clipping OOD** — dentro XGBoost ogni feature ha un segno imposto (più livello ⇒ mai peggio; più DPR nemico ⇒ mai meglio) e gli input vengono riportati nelle bande legali di 5e (AC 10–30, HP ≤ limiti, livello 1–20…). Un mostro homebrew da 10.000 HP degrada con grazia a "beyond deadly" invece di mandare in tilt gli alberi.
2. **Dominanza del roster** — negli incontri misti le medie pesate possono "diluire" (Lich + 6 goblin ha CR medio più basso del Lich da solo). Assioma: *aggiungere mostri non può mai rendere lo scontro più facile*. Il motore valuta anche ogni sotto-roster omogeneo e tiene il **minimo** delle probabilità.
3. **Guardia fisica di sopravvivenza** — la scoperta più importante del progetto: nei log reali persino gli scontri matematicamente senza speranza (party eliminato in ≤1 round) risultano "vinti" l'**84,5%** delle volte — è la *misericordia del DM* (dadi truccati, ritirate, rinforzi). Nessun modello addestrato su quei dati può rispondere a "e se combattessimo 19 Lich **fino alla morte**?". Perciò: `P(win) ≤ min( sigmoide(0,192 · round_di_sopravvivenza − 2,086 · ln(round_per_uccidere) + 4,122), lattice(CR, numero, livello) )` — tutto **calibrato su simulazione esterna di deathmatch**, niente costanti a mano. Il *race cap* è una regressione logistica pesata su entrambi i lati della corsa al danno (quanto sopravvivi, quanto ci metti a ucciderli — quest'ultimo non troncato, così un muro homebrew da 10.000 HP resta senza speranza), stimata sulle griglie Battlecast guard+OOD ricostruite con gli stessi profili del bestiario che l'app serve. Le versioni precedenti (a mano, poi un fit solo-sopravvivenza `sigmoide(1,63·s − 3,98)`) erano cieche alle sconfitte per logoramento: contro un Lich un party di livello 5 "sopravvive" 4+ round stimati, quindi il vecchio tetto restava a 0,95 mentre il simulatore dava al party **0,000** su 2.000 deathmatch — sopravvivere non è vincere quando 135 HP leggendari stanno dietro CA 17. E siccome due feature di TTK non possono prezzare una rotazione di incantesimi, dove la verità simulata esiste la serviamo direttamente: il *lattice* interpola trilinearmente la griglia guard stessa su (CR {2, 5, 10, 15, 21} × numero {1, 2, 4, 8, 12, 19} × livello {1, 5, 9, 13, 17, 20}), resa monotona in fase di build, e si astiene sotto CR 2 (`battlecast_bridge/guard_lattice.json`). Sulle celle della griglia la probabilità servita *è* quella simulata. Risultato: 1 Lich → livello **11** · 2 → **16,5** · **3+ → beyond deadly** (4 Lich: 7% a livello 20; 8+: 0,0%), in linea con i crossing al 65% del simulatore.

Accanto alla calibrazione sono state eseguite altre due griglie (in totale 304 celle, ~490.000 battaglie simulate, riproducibili con `make battlecast`):
- **Griglia "misericordia"** (25 mostri SRD da CR 0 a 10 × livelli {3, 7, 11, 15}, mostro singolo contro party bilanciato di 4 — una scansione sistematica, *non* il replay di incontri realmente registrati): confrontando le previsioni del modello (realtà del tavolo) con la verità deathmatch di Battlecast, il divario di misericordia risulta **bidirezionale**: il modello sta ~0,14 *sotto* il simulatore negli scontri facili (il tetto dell'83% di FIREBALL: ritirate ed etichette rumorose) e ~0,28 *sopra* negli scontri difficili (l'inflazione da misericordia del DM), correlazione 0,842 — `figures/battlecast_mercy_gap.png` (poche celle nei bin difficili: il +0,28 è indicativo, non preciso).
- **Griglia OOD** (cloni di un mostro CR 5 con HP ×{1, 5, 20} e AC +{0, 8}, 24 celle): il simulatore conferma i verdetti estremi — i cloni con HP ×20 vincono ≈ 0,000–0,003 dei deathmatch, coerente col nostro "beyond deadly" — mentre l'ordinamento fine di metà scala concorda solo moderatamente (Spearman 0,54).

---

## 6. Architettura del codice

```text
📁 project-dnd-fixed-v4
├── monster_offense.py      # Motore offensivo: parsing degli statblock SRD (DPR,
│                           #   attacco, DC, burst, multiattacco, incantesimi noti),
│                           #   tabelle DMG (CR↔XP, offense-by-CR, moltiplicatori),
│                           #   estrazione canonica dei tratti (unica per tutto il repo)
├── parse_fireball.py       # ETL: legge i log FIREBALL, aggancia bestiario+offense,
│                           #   aggrega a livello di incontro → clean_aggregated_combat_data.csv
├── initial_learn.py        # Pipeline ML: DnDFeatureEngineer (tutte le feature),
│                           #   tuning Optuna, CV raggruppata, calibrazione Platt
│                           #   group-aware, SHAP nativo → true_lethality_model.pkl
│                           #   + CR predictor (che CR darebbe WotC)
├── lethality_engine.py     # Nucleo d'inferenza condiviso: profili mostro, roster
│                           #   misti, ricerca binaria, curve, griglia 400 party,
│                           #   guardia di dominanza + guardia fisica,
│                           #   stima ufficiale DMG p.82 (official_encounter_estimate)
├── app.py                  # Interfaccia Streamlit (3 tab, grafici Plotly)
├── fair_fight_finder.py    # Gemello da terminale dell'app
├── model_comparison.py     # Benchmark del corso: logistica ridge, kernel RFF,
│                           #   test MMD a due campioni, Gaussian Process
├── behavior_suite.py       # 13 controlli comportamentali (assiomi di dominio):
│                           #   da eseguire dopo OGNI modifica al modello
├── tests/                  # 61 test pytest (regex di parsing, feature math,
│                           #   guardia fisica, matematica del manuale)
├── true_lethality_model.pkl            # pipeline calibrata (modello servito)
├── true_lethality_model_uncalibrated_xgb.json  # booster nativo (portabile tra versioni)
├── cr_predictor_model.{pkl,json}       # predittore del CR ufficiale
├── monster_offense_stats.csv           # tabella offensiva per-mostro (generata)
├── clean_aggregated_combat_data.csv    # dataset di addestramento (generato)
└── figures/                # metrics.json, SHAP, calibrazione, importanze, benchmark
```

Perché due interfacce non divergono mai: **tutta** la logica di simulazione vive in `lethality_engine.py`; `app.py` e `fair_fight_finder.py` la importano soltanto.

### Indice delle figure (`figures/`)

| Figura | Cosa mostra |
|---|---|
| `shap_summary.png` + `shap_ranking.csv` | SHAP nativo: quali feature guidano la P(win) e in che direzione |
| `feature_importance.png` | Importanza per gain di XGBoost |
| `calibration_curve.png` | Probabilità previste vs frequenze osservate su campagne held-out |
| `feature_correlation_heatmap.png` | Correlazioni di Spearman tra le feature (controllo collinearità) |
| `win_rate_vs_cr_ratio.png` | Tasso di vittoria empirico vs difficoltà normalizzata, con overlay del modello |
| `model_comparison.png` + `.csv` | Benchmark del corso vs modello di produzione (CV raggruppata) |
| `logistic_coefficients.png` | I coefficienti della logistica scartata — la "prova" del confondimento (livello del party col segno sbagliato) |
| `battlecast_guard_fit.png` | Griglia deathmatch + guardia calibrata vs quella scelta a mano |
| `battlecast_mercy_gap.png` | Modello (tavolo reale) vs simulatore (deathmatch): il divario di misericordia |
| `metrics.json` / `experiments.jsonl` / `course_benchmark.json` / `battlecast_summary.json` / `gan_ablation.json` | Risultati leggibili da macchina: ultima run, storico append-only, benchmark |

---

## 7. Come eseguire tutto

```bash
# 1. Ambiente (versioni bloccate = quelle con cui i .pkl sono stati addestrati)
pip install -r requirements.txt

# 2. (Opzionale) Rigenerare il dataset dai log FIREBALL (~2 min)
python3 parse_fireball.py

# In alternativa, l'intera pipeline con un comando solo:
#   make retrain   (dati -> training -> test -> behavior suite)
#   make help      (tutti i target disponibili)

# 3. (Opzionale) Riaddestrare i modelli
python3 initial_learn.py --trials 40 --train-cr-predictor   # con tuning Optuna (~5 min)
python3 initial_learn.py --no-tune                          # riusa gli iperparametri salvati

# 4. Avviare il sito
streamlit run app.py

# 5. Versione da terminale
python3 fair_fight_finder.py

# 6. Verifiche (da lanciare dopo OGNI modifica al modello)
python3 -m pytest tests/ -q      # 64 test unitari
python3 behavior_suite.py        # 13 assiomi comportamentali

# 7. Benchmark allineato al corso (logistica, kernel RFF, MMD, GP)
python3 model_comparison.py
```

---

## 8. Limiti e onestà scientifica

- **Il tetto dell'83%.** I DM curano gli incontri e i giocatori si ritirano da quelli persi: nei dati la vittoria è la norma. Per questo i mostri deboli saturano ("trivial") e il verdetto sostituisce un finto livello frazionario.
- **La misericordia del DM contamina proprio la coda che ci serve.** Da qui la guardia fisica (§5.3): è una scelta di dominio dichiarata, documentata e testata — non un parametro appreso.
- **AUC ≈ 0,65 non è un oracolo.** Il modello *ordina* bene gli incontri, ma il caso, le tattiche e gli oggetti magici non sono nei dati. Le probabilità vanno lette come stime calibrate, non profezie.
- **Il confronto col manuale è leale.** Il lato "book" esegue il vero calcolo del DMG (XP × moltiplicatore, aggiustamento per taglia del party): le discrepanze segnalate misurano il modello contro *il miglior sforzo* del manuale, non contro una sua caricatura.
- **Validazione raggruppata ovunque.** Split casuali gonfierebbero ogni numero riportato: il gap tra CV raggruppata e holdout *è* la dimostrazione del leakage, e riportarlo è una scelta, non un difetto.

---

*Statistical Machine Learning — Progetto Finale · Potenziato da XGBoost, calibrazione di Platt e 34.907 tiri di dado veri.* 🎲
