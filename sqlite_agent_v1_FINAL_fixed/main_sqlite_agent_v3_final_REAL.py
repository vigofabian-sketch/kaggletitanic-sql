import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class SQLiteAgent:
    def __init__(self, db_path="titanic.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.models = {}
        self.feature_columns = []
        
    def setup_database(self, train_df, test_df):
        self.cursor.execute("DROP TABLE IF EXISTS train_raw")
        self.cursor.execute("DROP TABLE IF EXISTS test_raw")
        self.cursor.execute("DROP TABLE IF EXISTS features")
        self.cursor.execute("DROP TABLE IF EXISTS predictions")
        self.cursor.execute("DROP TABLE IF EXISTS model_scores")
        self.cursor.execute("""
            CREATE TABLE model_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT,
                cv_mean REAL,
                cv_std REAL
            )
        """)
        
        train_df.to_sql('train_raw', self.conn, index=False, if_exists='replace')
        test_df.to_sql('test_raw', self.conn, index=False, if_exists='replace')
        self.conn.commit()
        
    def feature_engineering_sql(self):
        self.cursor.execute("DROP TABLE IF EXISTS features")
        self.cursor.execute("DROP TABLE IF EXISTS test_features")
        self.cursor.execute("""
            CREATE TABLE features AS
            SELECT 
                PassengerId,
                Survived,
                Pclass,
                Sex,
                COALESCE(Age, (SELECT AVG(Age) FROM train_raw WHERE Age IS NOT NULL)) as Age,
                SibSp,
                Parch,
                COALESCE(Fare, (SELECT AVG(Fare) FROM train_raw WHERE Fare IS NOT NULL)) as Fare,
                COALESCE(Embarked, 'S') as Embarked,
                CASE 
                    WHEN Name LIKE '%Master.%' THEN 'Master'
                    WHEN Name LIKE '%Mr.%' THEN 'Mr'
                    WHEN Name LIKE '%Mrs.%' THEN 'Mrs'
                    WHEN Name LIKE '%Miss.%' THEN 'Miss'
                    ELSE 'Rare'
                END as Title,
                (SibSp + Parch + 1) as FamilySize,
                CASE WHEN (SibSp + Parch) = 0 THEN 1 ELSE 0 END as IsAlone,
                CASE WHEN Cabin IS NULL OR Cabin = '' THEN 0 ELSE 1 END as HasCabin,
                SUBSTR(Cabin, 1, 1) as Deck,
                CASE 
                    WHEN Fare <= 7.91 THEN 0
                    WHEN Fare <= 14.454 THEN 1
                    WHEN Fare <= 31 THEN 2
                    ELSE 3
                END as FareBin,
                CASE 
                    WHEN Age <= 16 THEN 0
                    WHEN Age <= 32 THEN 1
                    WHEN Age <= 48 THEN 2
                    WHEN Age <= 64 THEN 3
                    ELSE 4
                END as AgeBin
            FROM train_raw
        """)
        
        self.cursor.execute("""
            CREATE TABLE test_features AS
            SELECT 
                PassengerId,
                Pclass,
                Sex,
                COALESCE(Age, (SELECT AVG(Age) FROM train_raw WHERE Age IS NOT NULL)) as Age,
                SibSp,
                Parch,
                COALESCE(Fare, (SELECT AVG(Fare) FROM train_raw WHERE Fare IS NOT NULL)) as Fare,
                COALESCE(Embarked, 'S') as Embarked,
                CASE 
                    WHEN Name LIKE '%Master.%' THEN 'Master'
                    WHEN Name LIKE '%Mr.%' THEN 'Mr'
                    WHEN Name LIKE '%Mrs.%' THEN 'Mrs'
                    WHEN Name LIKE '%Miss.%' THEN 'Miss'
                    ELSE 'Rare'
                END as Title,
                (SibSp + Parch + 1) as FamilySize,
                CASE WHEN (SibSp + Parch) = 0 THEN 1 ELSE 0 END as IsAlone,
                CASE WHEN Cabin IS NULL OR Cabin = '' THEN 0 ELSE 1 END as HasCabin,
                SUBSTR(Cabin, 1, 1) as Deck,
                CASE 
                    WHEN Fare <= 7.91 THEN 0
                    WHEN Fare <= 14.454 THEN 1
                    WHEN Fare <= 31 THEN 2
                    ELSE 3
                END as FareBin,
                CASE 
                    WHEN Age <= 16 THEN 0
                    WHEN Age <= 32 THEN 1
                    WHEN Age <= 48 THEN 2
                    WHEN Age <= 64 THEN 3
                    ELSE 4
                END as AgeBin
            FROM test_raw
        """)
        self.conn.commit()
        
    def encode_features(self):
        df = pd.read_sql("SELECT * FROM features", self.conn)
        test_df = pd.read_sql("SELECT * FROM test_features", self.conn)
        
        le_dict = {}
        cat_cols = ['Sex', 'Embarked', 'Title', 'Deck']
        
        for col in cat_cols:
            le = LabelEncoder()
            all_values = list(df[col].fillna('Unknown')) + list(test_df[col].fillna('Unknown'))
            le.fit(all_values)
            df[col] = le.transform(df[col].fillna('Unknown'))
            test_df[col] = le.transform(test_df[col].fillna('Unknown'))
            le_dict[col] = le
            
        self.feature_columns = [c for c in df.columns if c not in ['PassengerId', 'Survived']]
        
        df.to_sql('features_encoded', self.conn, index=False, if_exists='replace')
        test_df.to_sql('test_features_encoded', self.conn, index=False, if_exists='replace')
        self.conn.commit()
        
        return df, test_df
    
    def train_models(self, df):
        X = df[self.feature_columns]
        y = df['Survived']
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, 
                                    min_samples_leaf=2, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, 
                                        subsample=0.8, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        
        rf_scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')
        gb_scores = cross_val_score(gb, X, y, cv=skf, scoring='accuracy')
        lr_scores = cross_val_score(lr, X, y, cv=skf, scoring='accuracy')
        
        rf.fit(X, y)
        gb.fit(X, y)
        lr.fit(X, y)
        
        self.models = {'rf': rf, 'gb': gb, 'lr': lr}
        
        self.cursor.execute("""
            INSERT INTO model_scores (model, cv_mean, cv_std) VALUES 
            ('RandomForest', ?, ?),
            ('GradientBoosting', ?, ?),
            ('LogisticRegression', ?, ?)
        """, (rf_scores.mean(), rf_scores.std(), gb_scores.mean(), gb_scores.std(), 
              lr_scores.mean(), lr_scores.std()))
        self.conn.commit()
        
        print(f"RF CV: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
        print(f"GB CV: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")
        print(f"LR CV: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")
        
    def ensemble_predict(self, test_df):
        X_test = test_df[self.feature_columns]
        
        rf_pred = self.models['rf'].predict_proba(X_test)[:, 1]
        gb_pred = self.models['gb'].predict_proba(X_test)[:, 1]
        lr_pred = self.models['lr'].predict_proba(X_test)[:, 1]
        
        ensemble_pred = (rf_pred * 0.4 + gb_pred * 0.4 + lr_pred * 0.2)
        final_pred = (ensemble_pred > 0.5).astype(int)
        
        return final_pred, ensemble_pred
    
    def save_predictions(self, test_df, predictions, probabilities):
        pred_df = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': predictions,
            'Probability': probabilities
        })
        pred_df.to_sql('predictions', self.conn, index=False, if_exists='replace')
        
        submission = pred_df[['PassengerId', 'Survived']]
        submission.to_csv('submission_sqlite_agent.csv', index=False)
        print("Submission saved to submission_sqlite_agent.csv")
        
    def run(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print("Setting up database...")
        self.setup_database(train_df, test_df)
        
        print("Running feature engineering in SQL...")
        self.feature_engineering_sql()
        
        print("Encoding features...")
        df, test_df = self.encode_features()
        
        print("Training models...")
        self.train_models(df)
        
        print("Making predictions...")
        preds, probs = self.ensemble_predict(test_df)
        
        print("Saving results...")
        self.save_predictions(test_df, preds, probs)
        
        self.conn.close()
        return preds

if __name__ == "__main__":
    agent = SQLiteAgent()
    agent.run(
        "C:/Users/Consc/OneDrive/Documents/Kaggle/titanic/train.csv",
        "C:/Users/Consc/OneDrive/Documents/Kaggle/titanic/test.csv"
    )