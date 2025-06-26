PREDICTORS = ['annual_income',
        'monthly_inhand_salary',
        'num_bank_accounts',
        'num_credit_card',
        'interest_rate',
        'num_of_loan',
        'delay_from_due_date',
        'num_of_delayed_payment',
        'changed_credit_limit',
        'num_credit_inquiries',
        'outstanding_debt',
        'credit_utilization_ratio',
        'total_emi_per_month',
        'amount_invested_monthly',
        'monthly_balance',
        'credit_history_age_month',
        'fe_1','fe_2','fe_3','fe_4',
        'fe_5','fe_6','fe_7','fe_8','fe_9',
        'fe_10','fe_11','fe_12','fe_13','fe_14',
        'fe_15','fe_16','fe_17','fe_18','fe_19','fe_20'
        ]

PATH_DIR_DATA = ""
PATH_DIR_REPORT = "./report"

train_test_period_months = 12
oot_period_months = 2
train_test_ratio = 0.2

def create_model_artifact(best_model, transformer_stdscaler, config, 
                         X_train, X_test, X_oot, y_train, y_test, y_oot,
                         train_auc_score, test_auc_score, oot_auc_score, 
                         random_search):
    """Create model artifact dictionary from config template"""
    return {
        'model': best_model,
        'model_version': f"credit_model_{config['model_train_date_str'].replace('-','_')}",
        'preprocessing_transformers': {
            'stdscaler': transformer_stdscaler
        },
        'data_dates': config,
        'data_stats': {
            'X_train': X_train.shape[0],
            'X_test': X_test.shape[0],
            'X_oot': X_oot.shape[0],
            'y_train': round(y_train.mean(), 2),
            'y_test': round(y_test.mean(), 2),
            'y_oot': round(y_oot.mean(), 2)
        },
        'results': {
            'auc_train': train_auc_score,
            'auc_test': test_auc_score,
            'auc_oot': oot_auc_score,
            'gini_train': round(2*train_auc_score-1, 3),
            'gini_test': round(2*test_auc_score-1, 3),
            'gini_oot': round(2*oot_auc_score-1, 3)
        },
        'hp_params': random_search.best_params_
    }