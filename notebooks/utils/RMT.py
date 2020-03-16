import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class RMTClassifier(object):
    
    """
    Referred https://github.com/aced125/FCC_DNN_Ligand_Classification
    """
    
    def __init__(self,threshold_multiple = 1, cutoff = 0.95):
        self.cutoff = cutoff
        self.thresh_multiple = threshold_multiple
        

    def _RMT(self, matrix):
        N, p = matrix.shape
        
        gamma = p/N
        thresh = ((1 + np.sqrt(gamma))**2)*self.thresh_multiple
        
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)
        pca = PCA()
        pca.fit_transform(matrix)
        
        # Find significant vector space V
        dim_V = pca.explained_variance_[pca.explained_variance_>thresh].shape[0]
        
        return scaler, pca, dim_V
    
    def _distance_to_projection(self, data, vector_space):
        # Given a vector space, V, we project the data onto that vector space, and then return distance to
        # the original data
        
        # V is formatted such that different vectors are along different columns
        
        # Data matrix is formatted such that rows correspond to different data points
        # and columns correspond to different features
        
        # projected_matrix = D V V.T
        #
        # finally, subtract original data (projectect_matrix - D) take euclidean norm along columns 
        
        # Also note that the vector space can be as large/small as required
                
        return np.linalg.norm( np.dot(data, 
                                      np.dot( vector_space, vector_space.T ) )  - data,
                             axis = 1)
    
    def _distance(self, data, vector_space):
        return np.dot(data, np.dot(vector_space,vector_space.T)) - data
    
    def fit(self, X,y):
        
        X,y = np.array(X), np.array(y)
        actives = X[np.where(y==1)[0],:]
        inactives = X[np.where(y==0)[0],:]
        
        self.scaler_actives, self.pca_actives, self.dim_V_actives = self._RMT(actives)
        self.scaler_inactives, self.pca_inactives, self.dim_V_inactives = self._RMT(inactives)
        
        metric = self.predict_scores(actives)
        
        idx = np.argsort(metric) #sorts in ascending order
        metric = metric[idx]
        cutoff_idx = int(self.cutoff * len(metric))
        self.epsilon = metric[cutoff_idx]

    
    def predict_scores(self, X_test):
        # https://www.pnas.org/content/116/9/3373
        self.scores = (self._distance_to_projection(self.scaler_actives.transform(X_test), 
                                                 self.pca_actives.components_.T[:, :self.dim_V_actives])
                  - 
                  
                 self._distance_to_projection(self.scaler_inactives.transform(X_test), 
                                                 self.pca_inactives.components_.T[:, :self.dim_V_inactives]))
        return self.scores
    
    def projection(self, X):
        return self._distance(self.scaler_actives.transform(X), 
                                                 self.pca_actives.components_.T[:, :self.dim_V_actives])
    
    def predict(self, X_test, epsilon_multiple = 1):
        
        scores = self.predict_scores(X_test)
        predictions = np.array([1 if x<self.epsilon * epsilon_multiple else 0 for x in scores])
        
        return predictions
    
    def return_indices_of_common_molecules_of_active_eig(self, matrix, n=5,eigenvector_index=0):
        
        # returns molecules that lie closest to the selected eigenvector
        # default number of molecules returned = 5
        
        # Pick out the (best) eigenvector
        
        eig = self.feature_vecs[:,eigenvector_index].reshape(self.p,1)
        
        #Project molecules onto the one-dimensional vector space and get the indices of the top 5 molecules
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        return np.argpartition(    np.dot(matrix, eig).reshape(matrix.shape[0])   , -n)[-n:]