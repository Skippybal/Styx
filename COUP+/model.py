import numpy as np
import xgboost as xgb


class Model:
    """  """
    def __init__(self, u, feature_names, num_models=10, num_bootstrap_samples=1000, num_trees=2000, max_depth=2, learning_rate=.01, validation_configs=None, seed=1):
        self.u = u
        self.u_vect = np.vectorize(u, otypes=[float])
        self.feature_names = feature_names
        self.num_models = num_models
        self.num_bootstrap_samples = num_bootstrap_samples
        self.num_trees = num_trees
        self.model_params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'objective': 'reg:squarederror'}
        self.validation_configs = validation_configs
        np.random.seed(seed)


    def train(self, configs):
        print(f"Training model ... ")
        X = np.stack([configs[i]['config'].get_array() for i in configs])
        if type(configs[0]['runtimes']) is np.ndarray:
            y = np.array([np.mean(self.u_vect(configs[i]['runtimes'])) for i in configs])
        else:
            y = np.array([np.mean([self.u(t) for t in configs[i]['runtimes'].values()]) for i in configs])

        ## Take bootstrap samples and train models:
        self.models = []
        for i in range(self.num_models):
            sample = np.random.choice(y.shape[0], size=self.num_bootstrap_samples, replace=True)
            X_boot = X[sample, :]
            y_boot = y[sample]
            dboot = xgb.DMatrix(X_boot, y_boot, feature_names=self.feature_names)
            model = xgb.train(self.model_params, dboot, num_boost_round=self.num_trees)
            
            self.models.append(model)


    def predict(self, configs, average=True):
        if type(configs) == list:
            X_test = np.stack([c.get_array() for c in configs])
        elif type(configs) == dict:
            X_test = np.stack([configs[i]['config'].get_array() for i in configs])
        else: # just a single config
            X_test = [configs.get_array()]
        
        dtest = xgb.DMatrix(X_test, feature_names=self.feature_names)
        
        y_hats = np.zeros((len(self.models), len(configs)))
        for m, model in enumerate(self.models):
            y_hat = model.predict(dtest)
            y_hats[m, :] = y_hat
        if average:
            y_hats = np.mean(y_hats, axis=0)

        if type(configs) == list or type(configs) == dict:
            return y_hats
        else:
            return y_hats[0]



