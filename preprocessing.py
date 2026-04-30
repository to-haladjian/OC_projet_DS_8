import mlflow
import numpy as np
import pandas as pd
import gc

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from dataclasses import dataclass, asdict
from scipy import stats

from feature_validation import FeatureValidationConfig, validate_features, FeatureValidator, timer


############################################################
#                   Wrappers and Configs                   #
############################################################
@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Enable debug mode to use a subset of data
    debug: bool = False
    num_rows: int = 10000  # Used only if debug is True

    # NaN handling for different tables
    nan_as_category_train: bool = False
    nan_as_category_bureau_and_balance: bool = True
    nan_as_category_previous_application: bool = True
    nan_as_category_pos_cash: bool = True
    nan_as_category_installments: bool = True
    nan_as_category_credit_card: bool = True

    # Feature removal options
    validation_remove_high_correlation: bool = True
    validation_remove_low_variance: bool = True
    validation_remove_high_missing: bool = True
    validation_correlation_threshold: float = 0.90
    validation_variance_threshold: float = 0.01
    validation_missing_threshold: float = 0.5

    # Imputer and scaler settings
    imputer_strategy: str = "median"
    scaler_type: str = "StandardScaler"

    
    def to_dict(self):
        return asdict(self)
    
############################################################
#                       Tools functions                    #
############################################################

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    '''One-hot encode categorical features in the dataframe.'''
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Sanitize feature names by replacing special characters
def sanitize_feature_names(df):
    """Replace special characters in column names with underscores"""
    df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
    return df

############################################################
#                 Table processing fucntions               #
############################################################

# Preprocess application_train.csv and application_test.csv
def application_train(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('data/application_train.csv', nrows= num_rows)

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    df[cat_cols] = df[cat_cols].astype(int)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    new_features = pd.DataFrame({
        'DAYS_EMPLOYED_PERC': df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'],
        'INCOME_CREDIT_PERC': df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT'],
        'INCOME_PER_PERSON': df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'],
        'ANNUITY_INCOME_PERC': df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'],
        'PAYMENT_RATE': df['AMT_ANNUITY'] / df['AMT_CREDIT']
    }, index=df.index)
    
    df = pd.concat([df, new_features], axis=1)
    del new_features
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('data/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('data/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bb[bb_cat] = bb[bb_cat].astype(int)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    bureau[bureau_cat] = bureau[bureau_cat].astype(int)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('data/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    prev[cat_cols] = prev[cat_cols].astype(int)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('data/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    pos[cat_cols] = pos[cat_cols].astype(int)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('data/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    ins[cat_cols] = ins[cat_cols].astype(int)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('data/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    cc[cat_cols] = cc[cat_cols].astype(int)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


############################################################
#                    Join all tables                       #
############################################################

def join_all_data(config: PreprocessingConfig = None):
    """
    Join all data tables into a single dataframe.
    
    :param config: Description
    :type config: PreprocessingConfig
    """

    print(f"\n{'='*80}")
    print(f"Data Ingestion and Joining")
    print(f"{'='*80}")

    if config is None:
        config = PreprocessingConfig()
    
    num_rows = config.num_rows if config.debug else None
    
    steps = {
        "Processing bureau and bureau_balance...": (
            bureau_and_balance,
            "nan_as_category_bureau_and_balance",
        ),
        "Processing previous_applications...": (
            previous_applications,
            "nan_as_category_previous_application",
        ),
        "Processing POS-CASH balance...": (
            pos_cash,
            "nan_as_category_pos_cash",
        ),
        "Processing installments payments...": (
            installments_payments,
            "nan_as_category_installments",
        ),
        "Processing credit card balance...": (
            credit_card_balance,
            "nan_as_category_credit_card",
        ),
    }

    # Load train and test application data
    with timer():
        print(f"\n[1/{2+len(steps)}] Loading application data...")
        df = application_train(num_rows, 
                               nan_as_category= config.nan_as_category_train)
        print(f"  ✓ Dataset: {len(df)} samples, {len(df.columns)} features")

    for i, (label, (func, config_attr)) in enumerate(steps.items()):
        with timer():
            print(f"[{i+2}/{2+len(steps)}] {label}")
            aux_df = func(
                num_rows,
                nan_as_category=getattr(config, config_attr)
            )
            print(f"  ✓ Dataset: {len(aux_df)} samples, {len(aux_df.columns)} features")
            df = df.join(aux_df, how="left", on="SK_ID_CURR")
            del aux_df
            gc.collect()

    with timer():
        label = "Cleaning up feature names and infinite values..."
        print(f"[{2+len(steps)}/{2+len(steps)}] {label}")
        # Sanitize feature names
        df = sanitize_feature_names(df)
        # Replace inf values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        # Set SK_ID_CURR as index
        df = df.set_index('SK_ID_CURR')
        print(f"  ✓ Dataset after joining: {len(df)} samples, {len(df.columns)} features")
    return df

############################################################
#                Main preprocessing function               #
############################################################

# Full preprocessing pipeline
def preprocess_full_pipeline(config: PreprocessingConfig = None,
                             feature_validation_config: FeatureValidationConfig = None):
    """
    Full preprocessing pipeline without mixing train and test data.
    
    Returns:
        X_train, y_train, X_test, test_ids
    """
    if config is None:
        config = PreprocessingConfig()

    if feature_validation_config is None:
        feature_validation_config = FeatureValidationConfig()

    # Join all data
    df = join_all_data(config)

    # Save target and IDs
    y_train = df['TARGET'].astype(int)

    # Log dataset with MLflow
    train_df_mlflow = mlflow.data.from_pandas(df, name="home-credit-training", targets="TARGET")

    # Remove target
    train_df = df.drop(['TARGET'], axis=1, errors='ignore')
    del df
    gc.collect()

    feature_validator = FeatureValidator(config=feature_validation_config)
    validation_results = feature_validator.validate_all(train_df, y_train)
    
    train_df = feature_validator.get_cleaned_features(train_df)

    train_features = train_df.columns.tolist()



    with timer():
        print("\nPreprocessing: Imputation and Scaling...")
        # Instantiate imputer and scaler pipeline
        preprocess_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=config.imputer_strategy)),
            ("scaler", StandardScaler()),
        ])

        # Fit on the training data and transform training and test data
        train_transformed = preprocess_pipeline.fit_transform(train_df)

    # Save pipeline
    pipeline_path = "preprocessing_pipeline.pkl"
    joblib.dump(preprocess_pipeline, pipeline_path)

    train_df_transformed = pd.DataFrame(train_transformed, columns= train_features)

    print(f"Final train shape: {train_df_transformed.shape}")
    print(f"Target distribution: {np.bincount(y_train)}")
    
    # Return everything in a dict
    return {
        "train_data": train_df_transformed,
        "train_labels": y_train.values,
        "train_df_mlflow": train_df_mlflow,
        "pipeline": preprocess_pipeline,
        "pipeline_path": pipeline_path,
        "config": config,
        "feature_validation_config": feature_validation_config,
        "feature_validator": feature_validator,  # Return validator for MLflow logging
        "metadata": {
            "train_shape": train_transformed.shape,
            "n_features": train_transformed.shape[1],
            "class_distribution": np.bincount(y_train).tolist()
        }
    }

if __name__ == "__main__":
    config = PreprocessingConfig(debug= True, num_rows= 10000)
    preprocess_full_pipeline(config)