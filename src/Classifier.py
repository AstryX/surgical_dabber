from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import umap
import cv2
import numpy as np
import math
from joblib import dump
import time
import json
from Helper import readImagesAndMasks, extractColourFeatures
from Helper import extractNeighbourFeatures, extractMaskLabels


#Default parameter values
num_img = 41
fold_count = 5
num_label_data = 21400
percentage_data = 0.035
image_size_rows = 250
image_size_cols = 330
num_pixel_features = 54
one_pixel_features = 6
dim_red_features = 3
dim_red_features_neighbour = 18
hsv_v_tolerance = 30
hsv_v_index = 2
hsv_wrap_amount = 90

bool_add_neighbourhoods = True
bool_use_bayes = False
bool_use_label_ratios = False
bool_should_normalize = True
bool_should_dimensionally_reduce = True
bool_do_pca_experiment = True
bool_do_pca_separately = False
bool_exclude_border_pixels = True

pixel_neighbourhood_size = 3
neighbourhood_step = math.floor(pixel_neighbourhood_size / 2)
im_path = "./Dataset/Processed/"
classifier_name = "blood_classifier.joblib"
pca_name = "pca.joblib"
scaler_name = "scaler.joblib"
#~~~~~~~~~~~~~ Default values finish

param_name = "params.json"

with open(param_name) as json_data_file:
		
    data = json.load(json_data_file)
    
    if 'num_img' in data:
        num_img = data['num_img']
    if 'fold_count' in data:
        fold_count = data['fold_count']
    if 'num_label_data' in data:
        num_label_data = data['num_label_data']
    if 'percentage_data' in data:
        percentage_data = data['percentage_data']
    if 'image_size_rows' in data:
        image_size_rows = data['image_size_rows']
    if 'image_size_cols' in data:
        image_size_cols = data['image_size_cols']
    if 'num_pixel_features' in data:
        num_pixel_features = data['num_pixel_features']
    if 'one_pixel_features' in data:
        one_pixel_features = data['one_pixel_features']
    if 'dim_red_features' in data:
        dim_red_features = data['dim_red_features']
    if 'dim_red_features_neighbour' in data:
        dim_red_features_neighbour = data['dim_red_features_neighbour']
    if 'hsv_v_tolerance' in data:
        hsv_v_tolerance = data['hsv_v_tolerance']
    if 'hsv_v_index' in data:
        hsv_v_index = data['hsv_v_index']
    if 'hsv_wrap_amount' in data:
        hsv_wrap_amount = data['hsv_wrap_amount']
    if 'bool_add_neighbourhoods' in data:
        bool_add_neighbourhoods = data['bool_add_neighbourhoods']
    if 'bool_use_bayes' in data:
        bool_use_bayes = data['bool_use_bayes']
    if 'bool_use_label_ratios' in data:
        bool_use_label_ratios = data['bool_use_label_ratios']
    if 'bool_should_normalize' in data:
        bool_should_normalize = data['bool_should_normalize']
    if 'bool_should_dimensionally_reduce' in data:
        bool_should_dimensionally_reduce = data['bool_should_dimensionally_reduce']
    if 'bool_do_pca_experiment' in data:
        bool_do_pca_experiment = data['bool_do_pca_experiment']
    if 'bool_do_pca_separately' in data:
        bool_do_pca_separately = data['bool_do_pca_separately']
    if 'bool_exclude_border_pixels' in data:
        bool_exclude_border_pixels = data['bool_exclude_border_pixels']
    if 'pixel_neighbourhood_size' in data:
        pixel_neighbourhood_size = data['pixel_neighbourhood_size']
        neighbourhood_step = math.floor(pixel_neighbourhood_size / 2)
    if 'im_path' in data:
        im_path = data['im_path']
    if 'classifier_name' in data:
        classifier_name = data['classifier_name']
    if 'pca_name' in data:
        pca_name = data['pca_name']
    if 'scaler_name' in data:
        scaler_name = data['scaler_name']


def runUMAP(data, target):
    print('Testing UMAP dim. reduction!')
    reducer = umap.UMAP(n_neighbors=5, random_state=42)
    remap = reducer.fit(data)

    plt.scatter(remap.embedding_[:, 0], remap.embedding_[:, 1], s= 2, c=target, cmap='Spectral')
    plt.title('UMAP embedding of the data set', fontsize=24);

    plt.show()

def fitPCA(data, num_components):
    pca = PCA(n_components=num_components)
    pca_result = pca.fit(data)
    
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    return pca_result, sum(pca.explained_variance_ratio_)
    
if bool_add_neighbourhoods == True:
    dim_red_features = dim_red_features_neighbour
    
t = time.time()    
    
im_array, mask_images = readImagesAndMasks(im_path, num_img, False)
singular_features = extractColourFeatures(im_array, hsv_wrap_amount)
del im_array
image_features, inclusion_mask = extractNeighbourFeatures(singular_features, bool_add_neighbourhoods, True,
    one_pixel_features, pixel_neighbourhood_size, num_pixel_features, hsv_v_index,
    image_size_rows, image_size_cols, neighbourhood_step, hsv_v_tolerance, bool_exclude_border_pixels)
del singular_features
mask_labels, blood_data, nonblood_data, blood_labels, nonblood_labels = extractMaskLabels(mask_images, image_features, inclusion_mask)

print('Blood dimensions:')
print(blood_data.shape)
print('Non-blood dimensions:')
print(nonblood_data.shape)
print('Feature dimensions:')
print(image_features.shape)
print('Mask dimensions:')
print(mask_labels.shape)

del mask_images
del inclusion_mask
del image_features

np.random.shuffle(blood_data)
np.random.shuffle(nonblood_data)


blood_ratio = num_label_data
nonblood_ratio = num_label_data
if bool_use_label_ratios == True:
    blood_ratio = math.floor(len(blood_data) * percentage_data)
    nonblood_ratio = math.floor(len(nonblood_data) * percentage_data)
    print('New blood pixel ratio: ' + str(blood_ratio))
    print('New nonblood pixel ratio: ' + str(nonblood_ratio))
    
data_set = blood_data[:blood_ratio]
data_set = np.concatenate((data_set, nonblood_data[:nonblood_ratio]), axis=0)

data_set_y = blood_labels[:blood_ratio]
data_set_y = np.concatenate((data_set_y, nonblood_labels[:nonblood_ratio]), axis=0)

print('Selected data dimensions:')
print(data_set.shape)

#runUMAP(data_set, data_set_y)
#quit()

if bool_add_neighbourhoods == False:
    n_bins = 25
    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()

    colors = ['blue', 'red']
    labels = ['non-blood', 'blood']

    ax0.hist([(np.array((nonblood_data[:nonblood_ratio,3]))),(np.array((blood_data[:blood_ratio,3])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax0.legend(prop={'size': 10})
    ax0.set_title('Chromaticity Red histogram')

    ax1.hist([(np.array((nonblood_data[:nonblood_ratio,4]))),(np.array((blood_data[:blood_ratio,4])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax1.legend(prop={'size': 10})
    ax1.set_title('Chromaticity Green histogram')

    ax2.hist([(np.array((nonblood_data[:nonblood_ratio,5]))),(np.array((blood_data[:blood_ratio,5])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax2.legend(prop={'size': 10})
    ax2.set_title('Chromaticity Blue histogram')

    ax3.hist([(np.array((nonblood_data[:nonblood_ratio,0]))),(np.array((blood_data[:blood_ratio,0])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax3.legend(prop={'size': 10})
    ax3.set_title('HSV Hue histogram (wrapped by ' + str(hsv_wrap_amount) + ')')

    ax4.hist([(np.array((nonblood_data[:nonblood_ratio,1]))),(np.array((blood_data[:blood_ratio,1])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax4.legend(prop={'size': 10})
    ax4.set_title('HSV Saturation histogram')

    ax5.hist([(np.array((nonblood_data[:nonblood_ratio,2]))),(np.array((blood_data[:blood_ratio,2])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax5.legend(prop={'size': 10})
    ax5.set_title('HSV Value histogram')

    fig.tight_layout()
    plt.show()
    
del blood_data
del nonblood_data
del blood_labels
del nonblood_labels

print('Splitting into train and test sets!')
X_train, X_test, y_train, y_test = train_test_split(data_set, data_set_y, test_size=0.2, random_state=0)

del data_set
del data_set_y

print('Scaling data by removing mean and normalizing by std!')
#scaler = StandardScaler(with_std=False)
scaler = StandardScaler()
scaler = scaler.fit(X_train)
if bool_should_normalize == True:
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)
    
print('Applying linear PCA dimensionality reduction!')
print('Explained variances for defined number(default 18) out of 54 features:')
dim_red_model = None
dim_red_model_secondary = None
if bool_do_pca_separately == False:
    dim_red_model, perc_explained = fitPCA(X_train, dim_red_features)
    print(perc_explained)
else:
    half_feats = math.floor(one_pixel_features/2)
    dim_red_model, perc_explained = fitPCA(X_train[:,:half_feats], half_feats)
    dim_red_model_secondary, perc_explained_s = fitPCA(X_train[:,half_feats:], half_feats)
    print(perc_explained)
    print(perc_explained_s)
if bool_should_dimensionally_reduce == True:
    if bool_do_pca_separately == True:
        half_feats = math.floor(one_pixel_features/2)
        X_train_l = dim_red_model.transform(X_train[:,:half_feats])
        X_test_l = dim_red_model.transform(X_test[:, :half_feats])
        X_train_r = dim_red_model_secondary.transform(X_train[:, half_feats:])
        X_test_r = dim_red_model_secondary.transform(X_test[:, half_feats:])
        X_train = np.concatenate((X_train_l,X_train_r), axis=1)
        X_test = np.concatenate((X_test_l,X_test_r), axis=1)
        print(X_train.shape)
        print(X_test.shape)
    else:
        X_train = dim_red_model.transform(X_train)
        X_test = dim_red_model.transform(X_test)
    
#PCA experiment below:
if ((bool_do_pca_experiment == True) and (bool_add_neighbourhoods == False)):
    '''
    As for eigenvector display question.

    Assume that vector X encodes the data. This could be a 6 vector at a 
    given pixel.
    Eg: (cr,cg,cb,h,s,v) for a 1*1 pixel or 4*6 for a 2*2 neightbourhood, etc.
    Assume that X has dimension D.

    Make scatter matrix S = (X-mean(X))*(X-mean(X))'
    Do PCA on S and get the D eigenvalues e_i and eigenvectors V_i

    Make N displays, N <= D

    for i = 1 ... whatever
    for k = -3 : 1 : +3
      Y_k = mean(X) + k * V_i
      split Y_k into A_k (the cromaticity values) and B_k (the HSV values)
    end

    Display the 7 A_k and 7 B_k images  (either as pixels or as unfolded 
    back into an 2*, 3*3 image etc.)

    You will probably need to magnify the pixels to eg. 50*50.

     From the 7 you might see some sort of a trend as you go from -3 to +3.
    Eg. with one component, you might see it go from dark to light.
    This might mean that there is little useful info - just average brightness.

    Eigenvectors with low eigenvalues are probably just noise,
    so you could ignore these after PCA.'''
    
    mean_changed_images_hsv = []
    mean_changed_images_rgb = []
    eigenvectors = dim_red_model.components_
    data_mean = scaler.mean_
    data_mean[0] = data_mean[0] / 180
    data_mean[1] = data_mean[1] / 255
    data_mean[2] = data_mean[2] / 255
    print('Eigenvectors:')
    print(eigenvectors)
    for x in range(len(eigenvectors)):
        print('Nr. ' + str(x+1) + ' eigenvector:')
        print(eigenvectors[x])
    print('Mean Pixel:')
    print(data_mean)
    loop_size = one_pixel_features
    if bool_do_pca_separately == True:
        loop_size = math.floor(one_pixel_features/2)
    for i in range(loop_size):
        cur_ev_features_hsv = []
        cur_ev_features_rgb = []
        #print(str(i+1) + ' nr. eigenvector')
        #print(eigenvectors[i])
        for k in range(-6, 7): 
            hsv_feat = None
            if bool_do_pca_separately == True:
                hsv_feat = data_mean[:3] + k * 0.05 * eigenvectors[i]
            else:
                hsv_feat = data_mean[:3] + k * 0.05 * eigenvectors[i,:3]
            hsv_feat[0] = hsv_feat[0] * 180
            hsv_feat[1] = hsv_feat[1] * 255
            hsv_feat[2] = hsv_feat[2] * 255
            
            hsv_feat[0] = hsv_feat[0] - hsv_wrap_amount
            
            if hsv_feat[0] < 0:
                hsv_feat[0] = 180 + hsv_feat[0] % 180
            if hsv_feat[1] < 0:
                hsv_feat[1] = 0
            if hsv_feat[2] < 0:
                hsv_feat[2] = 0
            if hsv_feat[0] > 179:
                hsv_feat[0] = hsv_feat[0] % 180
            if hsv_feat[1] > 255:
                hsv_feat[1] = 255
            if hsv_feat[2] > 255:
                hsv_feat[2] = 255
            print('HSV feat:')
            print(hsv_feat)
            hsv_feat = cv2.cvtColor(np.array(([[hsv_feat]])).astype(np.uint8), cv2.COLOR_HSV2BGR)
            hsv_feat = hsv_feat[0][0]
            rgb_feat = None
            if bool_do_pca_separately == True:
                rgb_feat = (data_mean[3:] + k * 0.25 * (
                    dim_red_model_secondary.components_[i])) * 255
            else:
                rgb_feat = (data_mean[3:] + k * 0.25 * eigenvectors[i,3:]) * 255
            if rgb_feat[0] < 0:
                rgb_feat[0] = 0
            if rgb_feat[1] < 0:
                rgb_feat[1] = 0
            if rgb_feat[2] < 0:
                rgb_feat[2] = 0
            if rgb_feat[0] > 255:
                rgb_feat[0] = 255
            if rgb_feat[1] > 255:
                rgb_feat[1] = 255
            if rgb_feat[2] > 255:
                rgb_feat[2] = 255
            print('RGB feat:')
            print(rgb_feat)
            rgb_feat = cv2.cvtColor(np.array(([[rgb_feat]])).astype(np.uint8), cv2.COLOR_RGB2BGR);
            rgb_feat = rgb_feat[0][0]
            hsv_feat_array = []
            rgb_feat_array = []
            for x in range(50):
                temp_hsv = []
                temp_rgb = []
                for z in range(50):
                    temp_hsv.append(hsv_feat)
                    temp_rgb.append(rgb_feat)
                hsv_feat_array.append(temp_hsv)
                rgb_feat_array.append(temp_rgb)
            hsv_feat_array = np.array((hsv_feat_array))
            rgb_feat_array = np.array((rgb_feat_array))
            #hsv_feat_array = cv2.cvtColor(hsv_feat_array, cv2.COLOR_HSV2BGR);
            #rgb_feat_array = cv2.cvtColor(rgb_feat_array, cv2.COLOR_RGB2BGR);     
            cur_ev_features_hsv.append(hsv_feat_array)
            cur_ev_features_rgb.append(rgb_feat_array)
        mean_changed_images_hsv.append(cur_ev_features_hsv)
        mean_changed_images_rgb.append(cur_ev_features_rgb)
        
    
    for i in range(len(mean_changed_images_hsv)):
        hsv_disp = []
        rgb_disp = []
        for j in range(len(mean_changed_images_hsv[i])):
            if j == 0:
                hsv_disp = mean_changed_images_hsv[i][j]
                rgb_disp = mean_changed_images_rgb[i][j]
            else:
                hsv_disp = np.concatenate((hsv_disp, mean_changed_images_hsv[i][j]), axis=1)
                rgb_disp = np.concatenate((rgb_disp, mean_changed_images_rgb[i][j]), axis=1)
        final_disp = np.concatenate((hsv_disp, rgb_disp), axis=0)
        cv2.imshow('Eigenvector nr. '+ str(i+1) +' transformed mean pixels (top hsv, bottom chromaticity)'+ 
            'over 7 iterations (-3,3).', hsv_disp) 
        cv2.imshow('Eigenvector nr. '+ str(i+1) +' transformed mean pixels (top hsv, bottom chromaticity)'+ 
            'over 7 iterations rgb (-3,3).', rgb_disp) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if ((bool_add_neighbourhoods == False) and (bool_should_dimensionally_reduce == True)):
    reduced_blood = []
    reduced_nonblood = []
    for i in range(len(y_train)):
        if y_train[i] == 1:
            reduced_blood.append(X_train[i])
        else:
            reduced_nonblood.append(X_train[i])
    n_bins = 25
    fig, axes = plt.subplots(nrows=2, ncols=3)
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()

    colors = ['blue', 'red']
    labels = ['non-blood', 'blood']
    
    reduced_blood = np.array((reduced_blood))
    reduced_nonblood = np.array((reduced_nonblood))
    
    ax0.hist([(np.array((reduced_nonblood[:,0]))),(np.array((reduced_blood[:,0])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax0.legend(prop={'size': 10})
    ax0.set_title('First principal component diagram')

    ax1.hist([(np.array((reduced_nonblood[:,1]))),(np.array((reduced_blood[:,1])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax1.legend(prop={'size': 10})
    ax1.set_title('Second principal component diagram')

    ax2.hist([(np.array((reduced_nonblood[:,2]))),(np.array((reduced_blood[:,2])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax2.legend(prop={'size': 10})
    ax2.set_title('Third principal component diagram')
    
    ax3.hist([(np.array((reduced_nonblood[:,3]))),(np.array((reduced_blood[:,3])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax3.legend(prop={'size': 10})
    ax3.set_title('Fourth principal component diagram')
    
    ax4.hist([(np.array((reduced_nonblood[:,4]))),(np.array((reduced_blood[:,4])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax4.legend(prop={'size': 10})
    ax4.set_title('Fifth principal component diagram')
    
    ax5.hist([(np.array((reduced_nonblood[:,5]))),(np.array((reduced_blood[:,5])))], n_bins, density=True, histtype='bar', color=colors, label=labels)
    ax5.legend(prop={'size': 10})
    ax5.set_title('Sixth principal component diagram')

    fig.tight_layout()
    plt.show()
    
if ((bool_add_neighbourhoods == False) and (bool_do_pca_separately==True)):
    X_train = X_train[:,[0,1,3,5]]
    X_test = X_test[:,[0,1,3,5]]
    
clf = None  
  
if bool_use_bayes == False:
    #clf = svm.LinearSVC(C=3.0, random_state=0, verbose=True, tol=1e-5, class_weight='balanced')
    
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=6, class_weight='balanced', verbose=True)
    
    #clf = svm.SVC(C=3.0,kernel='rbf',verbose=True, class_weight='balanced')
    
    #n_estimators = 8 #Number of threads
    #clf = BaggingClassifier(svm.SVC(C=3.0,kernel='rbf', probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators)
else:
    clf = GaussianNB()

print('Computing ' + str(fold_count) + '-fold crossvalidation scores!')
scores = cross_val_score(clf, X_train, y_train, cv=fold_count, scoring='f1_macro', verbose=7, n_jobs=-1)
print('Fitting a fully trained model!')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Crossval. accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
print('Classifier classification score:')
print(clf.score(X_test, y_test))
print('Classifier confusion matrix:')
print(confusion_matrix(y_test, y_pred))


print('Full training process completed! Saving the model to a file...')
dump(clf, classifier_name) 
dump(dim_red_model, pca_name) 
dump(scaler, scaler_name) 
print('Model saved successfully!')
print('Program execution time: ' + str(time.time() - t) + ' seconds')
