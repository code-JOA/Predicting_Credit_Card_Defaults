def column_na_count_df(df):
    """
    Generates a dataframe with one column listing 
    every column in the DataFrame parameter (df),
    and another column with the count of na's in the column.
    """
    na_dict = {}
    for col in df.columns:
        na_count = df[col].isna().sum()
        na_dict.update({col: na_count})
    na_count_df = pd.DataFrame.from_dict(data=na_dict, 
                                         orient='index',
                                         columns=['na_count'])
    na_count_df.reset_index(inplace=True)
    na_count_df.rename(columns={'index': 'column'},
                       inplace=True)
    return na_count_df


def get_cat(df):
    """get list of cat features from df"""
    cat = []
    for x in df.columns:
        if df[x].dtypes == 'object':
            cat.append(x)
    return cat

#cat1 = get_cat(df)


def get_nom(df):
    """get nom features"""
    nom = []
    for x in df.columns:
        if df[x].dtypes != 'object':
            nom.append(x)
    return nom[2:] # no need for feature id and age but customise according to df

# nom1 = get_nom(train)


# def filla_null(df):
#     """simple function to fill up all null values in dataframe with median value"""
#     for col in df.columns :   
#         try:
#             median = df[col].median()
#             df[col] = df[col].fillna(value = median)
#         except:
#             continue
# df.columns = df.columns.fillna('median')  


def check_duplicates(df):
    """Simple function to find all the duplicates in the dataframe"""
    if df.duplicated().sum():
        print("There were {} duplicates and they have been removed".format(df.duplicated().sum()))
        df.drop_duplicates(inplace=True)
    else:
        print("You are all clear of duplicates")


def boxplot_variation(feature1 , feature2 , feature3 , width = 16):
    fig , ax1 = plt.subplots(ncols=1, figsize =(width , 6))
    s = sns.boxplot(ax = ax1 , x = feature1 , y = feature2 , hue = feature3 , 
                   data = df , palette = 'Blues' , showfliers = False)
    s.set_xticklabels(s.get_xticklabels() , rotation = 90)
    plt.show()


def CMatrix(CM, labels=['pay','default']):
    """Confusion Matrix"""
    df = pd.DataFrame(data=CM, index=labels, columns=labels)
    df.index.name='TRUE'
    df.columns.name='PREDICTION'
    df.loc['Total'] = df.sum()
    df['Total'] = df.sum(axis=1)
    return df


def base_func(element):
    """train and fit the model"""
    model = element()
    model.fit(X_train , y_train)
    
    """predict"""
    train_preds = model.predict(X_train)
    test_preds = model.predict(x_test)
    
    """evaluation"""
    train_accuracy = roc_auc_score(y_train , train_preds)
    test_accuracy = roc_auc_score(y_test , test_preds)
    
    print(str(element))
    print("--------------------------------------------")
    print(f"Training Accuracy: {(train_accuracy * 100) :.4}%")
    print(f"Test Accuracy : {(test_accuracy * 100) :.4}%")
    
    """Store accuracy in a new DataFrame"""
    score_logreg = [element , train_accuracy , test_accuracy]
    print("------------------------------------------------")
    models = pd.DataFrame([score_logreg])
    
    
    
def run_model2(model, X_train, y_train,X_test, y_test ):
    model.fit(X_train, y_train)

    """predict"""
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    """evaluate"""
    train_accuracy = roc_auc_score(y_train, train_preds)
    test_accuracy = roc_auc_score(y_test, test_preds)
    report = classification_report(y_test, test_preds)

    """print confusion matrix"""
    cnf_matrix = confusion_matrix(y_test , test_preds)
    print("Confusion Matrix:\n" , cnf_matrix)

    """print reports of the model accuracy"""
    print('Model Scores')
    print("------------------------")
    print(f"Training Accuracy: {(train_accuracy * 100):.4}%")
    print(f"Test Accuracy:     {(test_accuracy * 100):.4}%")
    print("------------------------------------------------------")
    print('Classification Report : \n', report)
    print("-----------------------------------------------------")
    print("Confusion Matrix:\n" , cnf_matrix)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    