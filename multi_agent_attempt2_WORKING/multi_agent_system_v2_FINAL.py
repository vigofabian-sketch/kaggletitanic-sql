import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class DataAgent:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.le_dict = {}
        
    def load_data(self, train_path, test_path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        return self
    
    def basic_clean(self):
        for df in [self.train_df, self.test_df]:
            df['Age'] = df['Age'].fillna(df['Age'].median())
            df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        return self
    
    def get_data(self):
        return self.train_df, self.test_df

class FeatureAgent:
    def __init__(self):
        self.le_dict = {}
        self.scaler = StandardScaler()
        self.selector = None
        
    def extract_title(self, name):
        if 'Master.' in name: return 'Master'
        elif 'Mr.' in name: return 'Mr'
        elif 'Mrs.' in name: return 'Mrs'
        elif 'Miss.' in name: return 'Miss'
        elif 'Dr.' in name: return 'Dr'
        elif 'Rev.' in name: return 'Rev'
        elif 'Col.' in name: return 'Col'
        elif 'Major.' in name: return 'Major'
        elif 'Mlle.' in name: return 'Miss'
        elif 'Ms.' in name: return 'Miss'
        elif 'Mme.' in name: return 'Mrs'
        elif 'Don.' in name: return 'Rare'
        elif 'Lady.' in name: return 'Rare'
        elif 'Countess.' in name: return 'Rare'
        elif 'Jonkheer.' in name: return 'Rare'
        elif 'Sir.' in name: return 'Rare'
        elif 'Capt.' in name: return 'Rare'
        return 'Rare'
    
    def engineer(self, train_df, test_df):
        for df in [train_df, test_df]:
            df['Title'] = df['Name'].apply(self.extract_title)
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            df['HasCabin'] = df['Cabin'].notna().astype(int)
            df['Deck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notna(x) else 'U')
            df['TicketPrefix'] = df['Ticket'].apply(lambda x: x.split()[0] if ' ' in x else 'NONE')
            df['NameLength'] = df['Name'].apply(len)
            df['FarePerPerson'] = df['Fare'] / df['FamilySize']
            
            df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 65, 100], labels=[0,1,2,3,4,5])
            df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3], duplicates='drop')
            
            df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']
            df['Title_Pclass'] = df['Title'] + '_' + df['Pclass'].astype(str)
        
        return train_df, test_df
    
    def encode(self, train_df, test_df):
        cat_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPrefix', 'Pclass_Sex', 'Title_Pclass']
        
        for col in cat_cols:
            le = LabelEncoder()
            combined = pd.concat([train_df[col], test_df[col]], axis=0).fillna('Unknown')
            le.fit(combined)
            train_df[col] = le.transform(train_df[col].fillna('Unknown'))
            test_df[col] = le.transform(test_df[col].fillna('Unknown'))
            self.le_dict[col] = le
            
        return train_df, test_df
    
    def select_features(self, train_df, test_df, target_col='Survived'):
        exclude = ['PassengerId', 'Name', 'Ticket', 'Cabin', target_col, 'AgeBin', 'FareBin']
        feature_cols = [c for c in train_df.columns if c not in exclude]
        
        X = train_df[feature_cols]
        y = train_df[target_col]
        
        self.selector = SelectKBest(f_classif, k=min(20, len(feature_cols)))
        X_selected = self.selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in self.selector.get_support(indices=True)]
        
        X_test = test_df[selected_features]
        
        return train_df[selected_features], y, X_test, selected_features

class ModelAgent:
    def __init__(self, name, model, params=None):
        self.name = name
        self.model = model(**params) if params else model()
        self.cv_score = 0
        self.oof_pred = None
        
    def train_cv(self, X, y, cv=5):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        self.cv_score = scores.mean()
        self.oof_pred = cross_val_predict(self.model, X, y, cv=skf, method='predict_proba')[:, 1]
        return self.cv_score
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

class EnsembleAgent:
    def __init__(self):
        self.agents = []
        self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
        self.weights = None
        
    def add_agent(self, agent):
        self.agents.append(agent)
        
    def train_all(self, X, y):
        print("Training individual agents...")
        for agent in self.agents:
            score = agent.train_cv(X, y)
            agent.fit(X, y)
            print(f"  {agent.name}: CV={score:.4f}")
            
        oof_matrix = np.column_stack([a.oof_pred for a in self.agents])
        
        self.meta_model.fit(oof_matrix, y)
        meta_pred = self.meta_model.predict(oof_matrix)
        meta_score = accuracy_score(y, meta_pred)
        print(f"  Stacking Meta-Model CV: {meta_score:.4f}")
        
        return meta_score
    
    def predict_proba(self, X):
        pred_matrix = np.column_stack([a.predict_proba(X) for a in self.agents])
        return self.meta_model.predict_proba(pred_matrix)[:, 1]
    
    def weighted_predict(self, X, weights=None):
        if weights is None:
            weights = np.array([a.cv_score for a in self.agents])
            weights = weights / weights.sum()
        
        pred_matrix = np.column_stack([a.predict_proba(X) for a in self.agents])
        weighted = np.average(pred_matrix, axis=1, weights=weights)
        return weighted

class MultiAgentSystem:
    def __init__(self):
        self.data_agent = DataAgent()
        self.feature_agent = FeatureAgent()
        self.ensemble_agent = EnsembleAgent()
        
    def setup_agents(self):
        self.ensemble_agent.add_agent(ModelAgent('RF', RandomForestClassifier, 
            {'n_estimators': 300, 'max_depth': 12, 'min_samples_split': 4, 
             'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}))
        
        self.ensemble_agent.add_agent(ModelAgent('GB', GradientBoostingClassifier,
            {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05,
             'subsample': 0.8, 'random_state': 42}))
        
        self.ensemble_agent.add_agent(ModelAgent('SVM', SVC,
            {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'random_state': 42}))
        
        self.ensemble_agent.add_agent(ModelAgent('KNN', KNeighborsClassifier,
            {'n_neighbors': 7, 'weights': 'distance'}))
        
        self.ensemble_agent.add_agent(ModelAgent('LR', LogisticRegression,
            {'max_iter': 1000, 'random_state': 42}))
        
        self.ensemble_agent.add_agent(ModelAgent('NB', GaussianNB, {}))
        
    def run(self, train_path, test_path):
        print("=== Multi-Agent Titanic System ===\n")
        
        print("[DataAgent] Loading data...")
        self.data_agent.load_data(train_path, test_path).basic_clean()
        train_df, test_df = self.data_agent.get_data()
        
        print("[FeatureAgent] Engineering features...")
        train_df, test_df = self.feature_agent.engineer(train_df, test_df)
        train_df, test_df = self.feature_agent.encode(train_df, test_df)
        X_train, y_train, X_test, features = self.feature_agent.select_features(train_df, test_df)
        
        print(f"[FeatureAgent] Selected {len(features)} features: {features}\n")
        
        print("[ModelAgents] Setting up specialized agents...")
        self.setup_agents()
        
        print("[EnsembleAgent] Training ensemble...")
        self.ensemble_agent.train_all(X_train, y_train)
        
        print("\n[EnsembleAgent] Generating predictions...")
        stacking_probs = self.ensemble_agent.predict_proba(X_test)
        weighted_probs = self.ensemble_agent.weighted_predict(X_test)
        
        stacking_pred = (stacking_probs > 0.5).astype(int)
        weighted_pred = (weighted_probs > 0.5).astype(int)
        
        sub_stacking = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': stacking_pred})
        sub_weighted = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': weighted_pred})
        
        sub_stacking.to_csv('submission_stacking.csv', index=False)
        sub_weighted.to_csv('submission_weighted.csv', index=False)
        
        print("Submissions saved: submission_stacking.csv, submission_weighted.csv")
        
        return stacking_pred, weighted_pred

if __name__ == "__main__":
    system = MultiAgentSystem()
    system.run(
        "C:/Users/Consc/OneDrive/Documents/Kaggle/titanic/train.csv",
        "C:/Users/Consc/OneDrive/Documents/Kaggle/titanic/test.csv"
    )