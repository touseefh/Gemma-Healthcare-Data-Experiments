# Gemma Healthcar Data Experiments

# Clinical Record Modeling with Gemma — MIMIC-IV Demo

This project investigates whether a small instruction-tuned Gemma model can learn the structure and selected statistical characteristics of hospital-admission records from the MIMIC-IV Clinical Database Demo.

The experiment converts relevant relational data into consolidated admission-level JSON records, fine-tunes the model using supervised fine-tuning (SFT) and Low-Rank Adaptation (LoRA), and evaluates generated records for structure, numerical characteristics, laboratory ranges, and exact duplication.

> **Scope:** This is an experimental modeling exercise using the MIMIC-IV Demo. The generated records are not validated clinical records and must not be used for diagnosis, treatment, or clinical decision-making.

## Project objective

The objective is to assess whether a fine-tuned small language model can produce new admission-level records that reflect selected structural and numerical characteristics of the source data.

The model is given one fixed instruction:

```text
Generate synthetic data.
```

It is expected to return a structured JSON record. No individual real admission is supplied as a generation example in the prompt.

## Dataset

Source: [MIMIC-IV Clinical Database Demo](https://physionet.org/content/mimic-iv-demo/)

The experiment uses the demo version of MIMIC-IV, not the full MIMIC-IV database.

| Dataset characteristic | Count |
|---|---:|
| Hospital admissions | 275 |
| Patients | 100 |

The source information is stored across multiple relational tables. For this experiment, relevant information was combined into one JSON record per hospital admission.

### Information represented in each admission record

| Attribute group | Information |
|---|---|
| Demographics | Age and gender |
| Hospital admission | Admission and discharge information |
| Length of stay | Hospital stay duration |
| ICU information | ICU admission and stay duration |
| Diagnoses | Diagnoses associated with the admission |
| Procedures | Medical procedures |
| Laboratory measurements | Selected laboratory results |
| Medications | Selected medication information |

This admission-level representation brings several types of information together in a single record. It does not mean that the original database is a single flat table.

## Data preparation

### Patient-level split

The 275 admissions were divided into training and held-out sets by patient:

| Split | Admissions |
|---|---:|
| Training | 249 |
| Held-out | 26 |
| **Total** | **275** |

The split was performed at patient level rather than by randomly assigning individual admissions. This keeps admissions from the same patient on one side of the split and reduces the risk of patient-level overlap between training and held-out data.

### Training example construction

Each of the 249 training admissions was represented eight times. For each representation, a random subset of laboratory measurements and medications was selected.

```text
249 training admissions × 8 representations
= 1,992 training examples
```

These 1,992 examples are not 1,992 independent admissions. They are eight representations of each of the 249 original training admissions. The variations expose the model to different selections of laboratory and medication information while retaining the underlying admission-level record.

## Model and fine-tuning

| Setting | Configuration |
|---|---|
| Base model | `unsloth/gemma-4-E2B-it` |
| Training method | Supervised fine-tuning (SFT) |
| Parameter-efficient method | LoRA |
| LoRA rank | 16 |
| Learning rate | `1e-4` |
| Effective batch size | 8 |
| Optimization steps | 249 |
| Epochs | 1 |
| GPU | Tesla T4 |
| Environment | Google Colab |
| Random seed | 3407 |

The model was loaded using Unsloth. LoRA trains a small set of adapter parameters while the original model weights remain frozen. The reported trainable parameter share is approximately 0.49%.

The model completed one epoch over the 1,992 prepared training examples, corresponding to approximately 249 optimization steps at an effective batch size of eight.

## Training format

Each training example contains the fixed instruction and one admission-level JSON response. The following is an illustrative outline of the record structure, not an actual patient record:

```text
Generate synthetic data.

{
  "age": 65,
  "gender": "F",
  "hospital_stay_days": 6,
  "icu_stay_days": 2,
  "diagnoses": [...],
  "procedures": [...],
  "lab_results": [...],
  "medications": [...]
}
```

The training setup uses response-only loss. The model is optimized to generate the JSON record, rather than to reproduce the instruction.

## Generation runs

After fine-tuning, the base Gemma model is loaded with the clinical-record LoRA adapter. The same fixed instruction is used for both reported runs.

| Run | Temperature | Reported sample |
|---|---:|---:|
| Run A | 1.0 | 10 records |
| Run B | 0.7 | 30 records |

Temperature changes the randomness of token sampling. A higher value generally allows more variation; a lower value generally makes token selection more predictable. It does not guarantee better similarity to the source data.

## Evaluation

The experiment evaluates several aspects of the generated records:

- **Structure:** whether records can be parsed and contain the expected fields.
- **Exact duplication:** whether generated records exactly match training or held-out admissions.
- **Aggregate numerical characteristics:** including age, hospital stay, ICU stay, diagnoses per admission, and medication counts.
- **ICU admission share:** comparison of the proportion of records with an ICU stay.
- **Laboratory ranges:** whether generated laboratory values fall within the observed minimum and maximum for the corresponding real measurements.
- **Clinical terminology and codes:** observed issues with diagnosis codes, procedure codes, and medication names.

Range compliance is only a basic check. A value falling between the observed minimum and maximum does not establish that it is clinically plausible or that it follows the source distribution.

## Reported results

The following figures are reported in the experiment notes.

### Aggregate characteristics

| Metric | Real data (275) | Run A (10) | Run B (30) |
|---|---:|---:|---:|
| Mean age | 62.73 | 66.10 | 62.80 |
| Mean hospital stay (days) | 6.88 | 7.87 | 5.98 |
| Mean ICU stay (days) | 3.63 | 5.62 | 1.60 |
| Mean diagnoses per stay | 16.39 | 17.90 | 16.67 |
| Mean medications | 31.08 | 37.30 | 30.73 |
| Share with ICU stay | 0.47 | 0.40 | 0.47 |

Run B was closer to the real-data mean for age, diagnoses per stay, medication count, and ICU admission share. It did not match every metric: the mean hospital stay was lower than the real-data mean, and the mean ICU stay was substantially lower.

### Run B validation checks

| Evaluation | Reported result |
|---|---:|
| Records passing validation | 32 of 32 |
| Records retained in reported sample | 30 |
| Generated laboratory values | 238 |
| Values within corresponding observed real-data min-max ranges | 238 |
| Exact copies of training or held-out admissions | 0 |

All 238 evaluated laboratory values fell within the corresponding observed real-data minimum and maximum ranges. This result establishes range compliance for the evaluated values only; it does not establish clinical validity or distributional equivalence.

### Observed code and terminology issues

The experiment notes report:

| Check | Result |
|---|---:|
| Diagnosis codes matching their associated titles in the real ICD dictionary | 18.8% |
| Procedure codes matching their associated titles | 0% |

Approximately half of the records also had inconsistencies between the number of diagnoses or procedures listed and their declared counts.

Some generated medication names were malformed or appeared to be invented. One observed example was:

```text
Glycombinant Adrenalin (Flagyl)
```

Medication names were not systematically checked against an authoritative medication database.

### Demographic and outcome differences

| Attribute | Real data | Generated sample |
|---|---:|---:|
| White patients | 62% | 87% |
| Death rate | 5.5% | 0% |

The generated sample did not reproduce these observed proportions. The absence of deaths in a sample of 30 may occur by chance, but it still means that this run did not represent the observed death rate.

## Limitations

- **Small evaluation sample:** Run B retained 30 records. This is too small for strong conclusions about rare clinical events or tail distributions.
- **Code-to-description consistency:** diagnosis and procedure codes were not reliably aligned with their associated titles.
- **Count inconsistencies:** some records had mismatches between listed diagnoses or procedures and their declared counts.
- **Medication naming:** some generated medication names were malformed or appeared invented; systematic validation against an authoritative source was not performed.
- **Demographic and outcome mismatch:** the generated sample differed from the real data for race and mortality proportions.
- **Range checks are limited:** laboratory values within observed minimum and maximum ranges are not necessarily clinically plausible.
- **Privacy is not established:** exact duplicate checks do not measure near-duplicate memorization, membership inference, or attribute disclosure.
- **Demo dataset scope:** the experiment uses the MIMIC-IV Demo, so its findings should not be generalized to the full MIMIC-IV database or broader hospital populations.

## Conclusion

The experiment shows that a small fine-tuned Gemma model can produce structured admission-level records and approximate some aggregate characteristics of the MIMIC-IV Demo data. In the reported 30-record run, several aggregate metrics were close to the real-data values, and all 238 evaluated laboratory values fell within their corresponding observed ranges.

However, the results also show important limitations. Diagnosis and procedure code matching was poor, medication names could be malformed, and demographic and outcome distributions differed from the source data. The experiment therefore demonstrates partial learning of record structure and selected numerical patterns, not clinically reliable record generation.
