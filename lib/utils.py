import pandas as pd
import sympy 
import pickle

## ML models
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb
import shap
from functools import cache

def normalise_sales(df):
    
    """
    Normalize sales for each year to its maximum value.    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'n', 'year', and 'sales'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized sales.
    """
    df = df.pivot(index = 'n', columns='year', values='sales')
    for col in df.columns:
        df[col] = df[col] / df[col].max()
    return df

def get_sales_stats(df): 
    df_aux = df.copy()

    df['median'] = df_aux.median(axis=1)
    df['mean']=df_aux.mean(axis=1)
    df['std'] = df_aux.std(axis=1)
    for col in df.columns:
        df[col] = df[col].fillna(df['median'])
    return df


def clean_features(features):
    features = features.drop(columns = ['is_odd', 'start_digit', 'n', 'str_n', 'leap_metric', 'odd_count', 'repeat_sum', 'has_repeated_digits','repeat_max', 'repeat_digit_count', 'dist_digits_count', 'ends_00', 'starts_00', 'is_prime', 'starts_15', 'ends_15'])
    features = pd.get_dummies(features, columns=['repeat_consec_max'], drop_first=True)
    return features

def compare_models(X_train, y_train, X_test, y_test, models = None):
    if models is None:
        models = {
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regressor': LinearRegression(),
            'KNN': KNeighborsRegressor(),
            'Bagging Regressor': BaggingRegressor(DecisionTreeRegressor(max_depth=20), n_estimators=100, max_samples = 1000),
            'Gradient Boosting Regressor' : GradientBoostingRegressor(n_estimators=100, max_depth = 7, learning_rate = 0.1),
            'AdaBoostRegressor': AdaBoostRegressor(DecisionTreeRegressor(max_depth=20), n_estimators=100),
            'xgb_reg': xgb.XGBRegressor(max_depth=20, n_estimators=100,
                                    learning_rate=0.1,  # default learning rate
                                    booster='gbtree',   # default booster
                                    objective='reg:squarederror')  # for regression tasks
            }
    # Train and evaluate models
    results = {}

    for name, model in models.items():
    # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        pred = model.predict(X_test)

        print(name)

        print("MAE", mean_absolute_error(pred, y_test))
        print("RMSE", root_mean_squared_error(pred, y_test))
        print("R2 score", model.score(X_test, y_test), "\n")

class Nr_properties(pd.DataFrame):

    def __init__(self):
        """
        Initialize the dataframe with the number properties.

        The dataframe is initialized with a range of numbers from 0 to 99999.
        Then, the methods starting with 'prop_' are applied to each number and
        the result is added to the dataframe with the name of the method as the
        column name. 
        The same is done for the methods starting with 'sprop_'
        but with the string representation of the number.
        """
        super().__init__({'n': list(range(100000))})

        self['str_n'] = self['n'].apply(lambda x: '0'*(5-len(str(x))) + str(x))

        nr_methods = [getattr(self, f) for f in dir(self) if f.startswith('prop')]
        for method in nr_methods:
            self[method.__doc__] = self['n'].apply(method)

        str_method = [getattr(self, f) for f in dir(self) if f.startswith('sprop_')]
        for method in str_method:
            self[method.__doc__] = self['str_n'].apply(method)

        postalcodes = pd.read_csv("../data/raw/codigos_postales.csv", dtype=str)
        postalcodes = set(postalcodes['codigo_postal'].values)
        self['is_postal_code'] = self['str_n'].apply(lambda x: x in postalcodes)

        
    # NUMERIC METHODS: 
    ## * ODD DIGITS PROPERTIES
    def prop_is_odd(self, n):
        '''is_odd'''
        return n % 2 == 1
    
    def prop_odd_count(self, n):
        '''odd_count'''
        return sum([int(nr) % 2 for nr in str(n)])

    ## * LAST DIGITS PROPERTIES
    def prop_ends_0(self, n):
        '''ends_0'''
        return n % 10 == 0

    def prop_ends_5(self, n):
        '''ends_5'''
        return n % 10 == 5

    def prop_ends_7(self, n):
        '''ends_7'''
        return n % 10 == 7
    
    def prop_ends_00(self, n):
        '''ends_00'''
        return n % 100 == 0
    def prop_ends_13(self, n):
        '''ends_13'''
        return n % 100 == 13
 
    def prop_ends_15(self, n):
        '''ends_15'''
        return n % 100 == 15
    
    def prop_ends_69(self, n):
        '''ends_69'''
        return n % 100 == 69
        
    ## * FIRST DIGIT PROPERTIES
    def prop_start_digit(self, n):
        '''start_digit'''
        return n // 10000

    def prop_starts_0(self, n):
        '''starts_0'''
        return n // 10000 == 0
    
    def prop_starts_9(self, n):
        '''starts_9'''
        return n // 10000 == 9
    
    def prop_starts_00(self, n):
        '''starts_00'''
        return n // 1000 == 0
    
    def prop_starts_13(self, n):
        '''starts_13'''
        return n // 1000 == 13
    
    def prop_starts_15(self, n):
        '''starts_15'''
        return n // 1000 == 15
    
    ## OTHER DIGIT PROPERTIES
    def prop_contains_13(self, n):
        '''contains_13'''
        return '13' in str(n)[1:4]
    
    ## * PRIME NUMBER PROPERTIES
    def prop_is_prime(self, n):
        '''is_prime'''
        return sympy.isprime(n)
    
    def prop_ends_prime(self, n):
        '''ends_prime'''
        return sympy.isprime(n % 100)
    
    ## STRING METHODS
    ## * REPEATED DIGITS PROPERTIES
    def sprop_has_repeated_digits(self, n):
        '''has_repeated_digits'''
        return len(set(n)) < len(n)
    
    def sprop_digits_count(self, n):
        '''dist_digits_count'''
        return len(set(n))
    
    def sprop_repeat_sum(self, n):
        '''repeat_sum'''
        return sum([n.count(x) for x in n])
    
    def sprop_repeat_max(self, n):
        '''repeat_max'''
        return max([n.count(x) for x in n])
    
    def sprop_repeat_digit_count(self, n):
        '''repeat_digit_count'''
        return len({x for x in n if n.count(x) > 1})
    
    def sprop_consecutive_repeat_max(self, n):
        '''repeat_consec_max'''
        pattern = [1]

        for i in range(1, len(n)):
            if n[i] == n[i - 1]:
                pattern[-1] += 1
            else:
                pattern.append(1)

        return max(pattern)
    
    ## PALINDROME
    def sprop_is_palindrome(self, n):
        '''is_palindrome'''
        return n == n[::-1] and len(set(n)) > 1 
    
    ## SERIES PROPERTIES: 
    def sprop_is_series(self, n):
        '''is_series'''
        serie = {int(n[i])-int(n[i-1]) for i in range(1,len(n))}
        return len(serie) == 1 and serie != {0}

    def sprop_leap_metric(self,n):
        '''leap_metric'''
        leaps = [abs(int(n[i])-int(n[i-1])) for i in range(1,len(n))]
        return sum(leaps)
    
    # EXTERNAL REFERENCE
    # IS DATE
    def sprop_is_date(self, n):
        '''is_date'''
        d, M = int(n[0]), int(n[1:3])
        if d > 0 and M > 0 and M < 13:
            return True
        
        d, M = int(n[:2]), int(n[3])
        if d > 0 and d < 31 and M > 0: 
            if M in (1, 3, 5, 7, 8):
                return True
            elif M in (4, 6, 9) and d < 30:
                return True
            elif M == 2 and d < 29:
                return True

        return False    
 
class Explainer():
    """
    Initialize the class with the necessary models and data.
    """
    def __init__(self):
        self.xgb_model = pickle.load(open('models/xgb_model.pkl', 'rb'))
        self.shap_model = pickle.load(open('models/shap_model.pkl', 'rb'))
        self.shap_values = pickle.load(open('models/shapvalues.sav', 'rb'))
        self.sales = pd.read_csv('data/lottery_nr_sales.csv')
        self.features = clean_features(pd.read_csv('data/nr_beauty_metrics.csv'))
        self.features_spanish = pd.DataFrame(index=self.features.columns, 
                                data={'spa': ['contiene el número 13', 'termina en 0', 'termina en 13', 'termina en 5', 'termina en 69', 'termina en 7', 'termina en número primo',
                                            'empieza en 0', 'empieza en 13', 'empieza en 9',
                                            'es una fecha', 'es palíndromo', 'es un código postal', 'es una serie',
                                            'tiene un dígito repetido 2 veces seguidas', 'tiene un dígito repetido 3 veces seguidas', 'tiene un dígito repetido 4 veces seguidas','tiene un dígito repetido 5 veces seguidas']
                                     })


    def beauty_features_explain(self, n):
        df = pd.DataFrame(index=self.features.columns, data={'coef': self.shap_values.values[n]*100, 'values': self.shap_values.data[n]})
        # sort values by coef and select only above 5 %
        top = df[df['coef'] > 5].sort_values(by='coef', ascending=False)
        if len(top) > 0:
            print(f'Estas son algunas características interesantes del número {n}')
            for i in range(len(top)):
                if top['values'].iloc[i] == False:
                    a = 'no '
                else:
                    a = ''
                if top['coef'].iloc[i] < 0:
                    b = 'disminuye'
                else:
                    b = 'aumenta'
                print (f'- {a}{self.features_spanish.loc[top.index[i], 'spa']}. Esto {b} las ventas esperadas en un {top['coef'].iloc[i]:.2f} %')
                
    def beauty_features_plot(self, n):
        shap.plots.waterfall(self.shap_values[n])

    def beauty_rating(self,n): 
        predicted = self.xgb_model.predict(self.features)[n]*100
        expected = self.shap_values.base_values[0]*100
        past_sales = self.sales['mean'].iloc[n]
        std_dev = self.sales['std'].iloc[n]

        str = ""

        if past_sales == 1:
            str += 'Este número es una reina de la belleza. Ha vendido todas las series de los últimos años\n'
        elif predicted > 93 and std_dev < 0.15:
            str += 'Este número es muy, pero que muy bonito\n'
        elif predicted > expected and std_dev < 0.20:
            str+='Este número es bastante bonito, y se espera que obtenga más ventas que la media\n'
        elif predicted < 0.15 and std_dev < 0.15: 
            str+='Este número es un adefesio. La gente no lo quiere ni regalao\n'
        elif predicted < (expected - 0.2) and std_dev < 0.2:
            str+='Este número es más bien feo. Se espera que obtenga menos ventas que la media\n'
        elif std_dev > 0.3:
            str+='Este número es un locurón. A veces vende mucho, otras muy poco... no sabemos muy bien por dónde pillarlo\n'
        else: 
            str+='Este número es del montón\n' 

        str+=f'Ha obtenido una puntuación de {predicted:.2f} sobre 100.'
        return str




def load_data():
    data = pd.read_csv("../data/venta_por_nr.csv")
    return data
 
def get_summary(data, nr):
    return data[data['n'] == nr]