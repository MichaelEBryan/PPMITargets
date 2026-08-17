#!/usr/bin/env python
"""
Leakage attribution for the published PPMI EOPD/LOPD classifier.

Question: what fraction of the reported classification performance
(weighted F1 0.95-0.96, macro OvR AUC 0.89-0.99) is carried by
ascertainment/design variables rather than by genotype?

Arms
  as_published   exactly the published preprocessing (all numeric cols, fillna(0),
                 single 80:20 split, seed 42)
  full_cv        same feature set, repeated stratified 5-fold CV
  -<block>       full feature set minus one block
  <block>_only   one block alone
  missingness    binary is-missing indicators only, no values at all

Blocks are defined by provenance, not by name-matching on the outcome.
"""
import os, re, json, sys, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import paths as _P

SRC = str(_P.DATA / "Complete DataFrameC.csv")
OUT = str(_P.ROOT / "results")
os.makedirs(OUT, exist_ok=True)
CLASSES = ['Healthy Control', "Parkinson's Disease", "Prodromal", "SWEDD"]

# Feature block definitions -- by data provenance
BLOCKS = {
    # PPMI enrolment / study-design variables. These encode which sub-study a
    # participant was recruited into, which is by construction near-collinear
    # with COHORT_DEFINITION.
    "design": [
        r"^ENRL", r"STDY$", r"^PPMI_ONLINE_ENROLL$", r"^SCREENEDAM$",
        r"^ENROLL_STATUS$", r"^DATELIG$", r"^APPRDX$", r"^COHORT_y$",
        r"^INEXPAGE$", r"^EVENT_ID", r"^REC_ID", r"^VISNO", r"^PAG_NAME",
        r"^ORIG_ENTRY", r"^LAST_UPDATE", r"^INFODT", r"^SE_(REC_ID|EVENT_ID|PAG_NAME|INFODT|ORIG_ENTRY|LAST_UPDATE)",
        r"^AM_(REC_ID|EVENT_ID|SUB_EVENT_ID|PAG_NAME|INFODT)", r"^MG_EVENT_ID$",
    ],
    # Genetics-core carrier calls + which assays were run on the sample.
    # Assay availability is itself an ascertainment variable in PPMI.
    "genetics_core": [
        r"^CLIA$", r"^GWAS$", r"^WES$", r"^WGS$", r"^SVs$", r"^SANGER$",
        r"^IU_Fingerprint$", r"^RNASEQ", r"^APOE$", r"^PATHVAR_COUNT$",
        r"^VAR_GENE$", r"^LRRK$", r"^GBA$", r"^VPS$", r"^SNCA$", r"^PRKN$",
        r"^PARK$", r"^PINK$", r"^NOTES$",
    ],
    # Digital gait / wearable / posturography. Motor phenotype = the thing the
    # diagnosis is made on.
    "motor": [
        r"^TUG", r"^SW_", r"^STEP_", r"^STR_", r"^SP_", r"^TRA_", r"^T_AMP",
        r"^L_JERK", r"^R_JERK", r"^JERK_T", r"^LA_", r"^RA_", r"^CAD_",
        r"^ASA_", r"^ASYM_IND", r"^SYM_", r"^GAIT_SUBGROUP",
        r"^(CV|Cadence|Degrees|Velocity|Time|Number|Mean|Sum|Total|Percent|Step|Walk|Wake|Lying|Sitting|Standing|Sedentary|Other|Night|Actual|Activity|Valid|Start|Stop|UpSideDown|NonWear)",
        r"^(amp|rms|wd|str|stp|step|stride|samp|Samp)", r"^x_", r"^x__",
    ],
    "family":  [r"PD$", r"^ANYFAMPD$", r"^BIOMOM", r"^BIODAD", r"^FUL", r"^HAFSIB",
                r"^MAGPAR", r"^PAGPAR", r"^MATAU", r"^PATAU", r"^MATCOUS", r"^PATCOUS",
                r"^MAHAFSIB", r"^PAHAFSIB", r"^KIDS", r"^DISFAMPD$"],
    "demo":    [r"^SEX$", r"^RA(ASIAN|BLACK|HAWOPI|INDALS|NOS|UNKNOWN|WHITE)$",
                r"^HISPLAT$", r"^HANDED$", r"^HOWLIVE$", r"^CHLDBEAR$", r"^BIRTHDT$",
                r"^(AFICBERB|ASHKJEW|BASQUE)$", r"^SE_EDUCYRS", r"^SE_Education",
                r"^(GAYLES|HETERO|BISEXUAL|PANSEXUAL|ASEXUAL|OTHSEXUALITY)$"],
    "age":     [r"^ENROLL_AGE$", r"^AGE_AT_VISIT"],
    "geno":    [r"^chr[0-9XYM]+:", r"^MG_rs"],
    "prs":     [r"^MG_Genetic_PRS_PRS"],
    "pcs":     [r"^MG_Genetic_PRS_PC", r"^MG_Genetic_PRS_InfPop$"],
}
DROP_ALWAYS = ['Unnamed: 0', 'Unnamed: 0.1', 'PATNO', 'ENROLL_DATE', 'STATUS_DATE',
               'COHORT_x', 'COHORT_DEFINITION']


def assign_blocks(cols):
    out, unassigned = {}, []
    for c in cols:
        hit = None
        for b, pats in BLOCKS.items():
            if any(re.search(p, c) for p in pats):
                hit = b
                break
        if hit is None:
            unassigned.append(c)
        out.setdefault(hit or "other", []).append(c)
    return out, unassigned


def models(seed=42, nclass=4):
    return {
        "XGBoost": xgb.XGBClassifier(objective='multi:softprob', n_estimators=300,
                                     max_depth=4, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, reg_lambda=1.0,
                                     random_state=seed, n_jobs=4, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(objective='multiclass', n_estimators=300,
                                       max_depth=4, learning_rate=0.05, subsample=0.8,
                                       colsample_bytree=0.8, reg_lambda=1.0,
                                       random_state=seed, n_jobs=4, verbose=-1),
        "CatBoost": CatBoostClassifier(loss_function='MultiClass', iterations=300,
                                       depth=4, learning_rate=0.05, l2_leaf_reg=3.0,
                                       random_seed=seed, verbose=0, thread_count=4,
                                       allow_writing_files=False),
    }


def evaluate(X, y, name, n_repeats=5, n_splits=5, seed=0):
    """Repeated stratified CV. Returns per-fold metric rows."""
    rows = []
    present = np.unique(y)
    if len(present) < 2 or min(np.bincount(y)[present]) < n_splits:
        return rows
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for fold, (tr, te) in enumerate(rskf.split(X, y)):
        if len(np.unique(y[tr])) < len(present):
            continue
        for mname, m in models(seed=fold).items():
            try:
                m.fit(X.iloc[tr], y[tr])
                p = m.predict_proba(X.iloc[te])
                yp = p.argmax(1)
                try:
                    auc = roc_auc_score(y[te], p, multi_class='ovr', average='macro',
                                        labels=present)
                except Exception:
                    auc = np.nan
                rows.append(dict(arm=name, model=mname, fold=fold,
                                 n_train=len(tr), n_test=len(te), n_feat=X.shape[1],
                                 f1w=f1_score(y[te], yp, average='weighted'),
                                 f1m=f1_score(y[te], yp, average='macro'),
                                 bacc=balanced_accuracy_score(y[te], yp),
                                 acc=accuracy_score(y[te], yp), auc=auc))
            except Exception as e:
                rows.append(dict(arm=name, model=mname, fold=fold, error=str(e)[:120]))
    return rows


def main():
    t0 = time.time()
    df = pd.read_csv(SRC, low_memory=False)
    print(f"loaded {df.shape}", flush=True)

    # 0. Is COHORT_DEFINITION a function of the ENRL* flags?
    enrl = [c for c in df.columns if c.startswith('ENRL')]
    sub = df[df.COHORT_DEFINITION.isin(CLASSES)].copy()
    key = sub[enrl].fillna(-1).astype(int).astype(str).agg('|'.join, axis=1)
    tab = pd.crosstab(key, sub.COHORT_DEFINITION)
    purity = tab.max(1).sum() / tab.values.sum()
    det = dict(n_enrl_flags=len(enrl), n_distinct_patterns=int(tab.shape[0]),
               n_participants=int(tab.values.sum()),
               majority_class_purity_of_enrl_pattern=float(purity),
               interpretation=("fraction of participants correctly labelled by simply "
                               "predicting the majority COHORT_DEFINITION of their "
                               "ENRL* flag pattern"))
    print("ENRL* -> COHORT_DEFINITION purity:", round(purity, 4), flush=True)
    json.dump(det, open(f"{OUT}/00_enrl_determinism.json", "w"), indent=2)
    tab.to_csv(f"{OUT}/00_enrl_pattern_table.csv")

    # 1. Exactly as published
    pub_rows = []
    for stratum, mask in [("EOPD(<50)", df.ENROLL_AGE < 50), ("LOPD(>=60)", df.ENROLL_AGE >= 60)]:
        d = df.drop(columns=[c for c in ['Unnamed: 0', 'PATNO', 'ENROLL_DATE',
                                         'STATUS_DATE', 'Unnamed: 0.1', 'COHORT_x',
                                         'COHORT_y'] if c in df.columns])
        d = d[d.COHORT_DEFINITION.isin(CLASSES)]
        d = d[mask.reindex(d.index).fillna(False)]
        X = d.drop(columns=['COHORT_DEFINITION']).select_dtypes(include=['int64', 'float64']).fillna(0)
        y = LabelEncoder().fit_transform(d['COHORT_DEFINITION'])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        for mname, m in models().items():
            Xtr2, Xte2 = Xtr.copy(), Xte.copy()
            if mname == "LightGBM":
                Xtr2.columns = [re.sub(r'[^\w]', '_', c) for c in Xtr2.columns]
                Xte2.columns = Xtr2.columns
            m.fit(Xtr2, ytr)
            p = m.predict_proba(Xte2); yp = p.argmax(1)
            pub_rows.append(dict(stratum=stratum, model=mname, n=len(d), n_feat=X.shape[1],
                                 f1w=f1_score(yte, yp, average='weighted'),
                                 f1m=f1_score(yte, yp, average='macro'),
                                 bacc=balanced_accuracy_score(yte, yp),
                                 auc=roc_auc_score(yte, p, multi_class='ovr', average='macro')))
            print(f"as-published {stratum:12s} {mname:9s} f1w={pub_rows[-1]['f1w']:.3f} "
                  f"auc={pub_rows[-1]['auc']:.3f}", flush=True)
    pd.DataFrame(pub_rows).to_csv(f"{OUT}/01_as_published.csv", index=False)

    # 2. Block ablation under repeated CV
    numcols = [c for c in df.select_dtypes(include=['int64', 'float64']).columns
               if c not in DROP_ALWAYS]
    blocks, unassigned = assign_blocks(numcols)
    json.dump({k: len(v) for k, v in blocks.items()}, open(f"{OUT}/02_block_sizes.json", "w"), indent=2)
    open(f"{OUT}/02_unassigned_cols.txt", "w").write("\n".join(unassigned))
    print("block sizes:", {k: len(v) for k, v in blocks.items()}, flush=True)
    print("unassigned:", len(unassigned), flush=True)

    all_rows = []
    for stratum, mask in [("EOPD(<50)", df.ENROLL_AGE < 50),
                          ("LOPD(>=60)", df.ENROLL_AGE >= 60),
                          ("ALL", pd.Series(True, index=df.index))]:
        d = df[df.COHORT_DEFINITION.isin(CLASSES) & mask.fillna(False)].copy()
        y = LabelEncoder().fit_transform(d['COHORT_DEFINITION'])
        Xall = d[numcols].copy()
        Xall.columns = [re.sub(r'[^\w]', '_', c) for c in Xall.columns]
        colmap = dict(zip(numcols, Xall.columns))
        Xfill = Xall.fillna(0)

        arms = {"full": list(Xall.columns)}
        for b, cols in blocks.items():
            cc = [colmap[c] for c in cols if c in colmap]
            if not cc:
                continue
            arms[f"minus_{b}"] = [c for c in Xall.columns if c not in set(cc)]
            arms[f"only_{b}"] = cc
        # the defensible genetic-only model
        gcols = [colmap[c] for b in ("geno", "prs", "pcs") for c in blocks.get(b, []) if c in colmap]
        if gcols:
            arms["genetics_only"] = gcols
            arms["genetics_plus_demo_age"] = gcols + [colmap[c] for b in ("demo", "age")
                                                      for c in blocks.get(b, []) if c in colmap]

        for arm, cols in arms.items():
            if not cols:
                continue
            rows = evaluate(Xfill[cols], y, f"{stratum}|{arm}")
            all_rows += rows
            ok = [r for r in rows if 'f1w' in r]
            if ok:
                print(f"  {stratum:12s} {arm:26s} nfeat={len(cols):4d} "
                      f"f1w={np.mean([r['f1w'] for r in ok]):.3f} "
                      f"auc={np.nanmean([r['auc'] for r in ok]):.3f}", flush=True)

        # missingness-only: binary indicators, no values
        miss = Xall.isna().astype(int)
        miss = miss.loc[:, miss.nunique() > 1]
        if miss.shape[1]:
            rows = evaluate(miss, y, f"{stratum}|missingness_only")
            all_rows += rows
            ok = [r for r in rows if 'f1w' in r]
            print(f"  {stratum:12s} {'missingness_only':26s} nfeat={miss.shape[1]:4d} "
                  f"f1w={np.mean([r['f1w'] for r in ok]):.3f} "
                  f"auc={np.nanmean([r['auc'] for r in ok]):.3f}", flush=True)

        # majority-class baseline
        base = []
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
        for tr, te in rskf.split(Xfill, y):
            dm = DummyClassifier(strategy="prior").fit(Xfill.iloc[tr], y[tr])
            yp = dm.predict(Xfill.iloc[te])
            base.append(dict(arm=f"{stratum}|majority_baseline", model="Dummy", fold=0,
                             f1w=f1_score(y[te], yp, average='weighted'),
                             f1m=f1_score(y[te], yp, average='macro'),
                             bacc=balanced_accuracy_score(y[te], yp),
                             acc=accuracy_score(y[te], yp), auc=0.5,
                             n_feat=0, n_train=len(tr), n_test=len(te)))
        all_rows += base
        print(f"  {stratum:12s} {'majority_baseline':26s} "
              f"f1w={np.mean([r['f1w'] for r in base]):.3f}", flush=True)

    res = pd.DataFrame(all_rows)
    res.to_csv(f"{OUT}/03_ablation_folds.csv", index=False)
    summ = (res[res.f1w.notna()].groupby(['arm', 'model'])
            .agg(n_feat=('n_feat', 'first'),
                 f1w_mean=('f1w', 'mean'), f1w_sd=('f1w', 'std'),
                 f1m_mean=('f1m', 'mean'), bacc_mean=('bacc', 'mean'),
                 auc_mean=('auc', 'mean'), auc_sd=('auc', 'std'),
                 folds=('f1w', 'size')).reset_index())
    summ.to_csv(f"{OUT}/03_ablation_summary.csv", index=False)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
