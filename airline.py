import pandas as pd

airline_df = pd.read_csv("data_airline_reviews.csv")
airline_df.info()

print("Dataset Shape (Rows, Columns):", airline_df.shape)

airline_df.columns

#unique values of the recommended column(target variable)
airline_df.recommended.unique()

# the unique value
airline_df.nunique()

#counting  number values
airline_df.duplicated().sum()

#droping the null
airline_df.drop_duplicates(inplace = True)

airline_df.duplicated().sum()

#Checking the null value count for each column
airline_df.isnull().sum()

airline_df.describe()

import matplotlib.pyplot as plt
import seaborn as sns

airline_df.dropna(subset=['traveller_type'],inplace=True)
plt.figure(figsize=(3,3))
sns.countplot(x=airline_df['traveller_type'],data=airline_df)
plt.xticks(rotation=90)
plt.show()

print(" ")

label_for_traveller = ['Solo Leisure','Couple Leisure','Family Leisure','Business']
data1 = airline_df['traveller_type'].value_counts().values
explode = [0.1, 0, 0, 0]
plt.figure(figsize=(4,4))
plt.pie(data1, labels = label_for_traveller,explode=explode,radius=1.5,autopct='%0.2f%%',shadow=True,textprops={'fontsize': 10})
plt.legend(loc='center',shadow=True,fancybox=True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
airline_df.dropna(subset=['recommended'],inplace=True)
plt.figure(figsize=(3,3))
sns.countplot(x=airline_df['cabin'],hue=airline_df['recommended'])
plt.xticks(rotation=90)
plt.show()

print(" ")


label_for_cabin = ['Economy Class','Business Class','Premium Economy','First Class']
data2 = airline_df['cabin'].value_counts().values
plt.figure(figsize=(4,4))
plt.pie(data2, labels = label_for_cabin,radius=1.5,autopct='%0.2f%%',shadow=True,textprops={'fontsize': 14})
plt.legend(loc='upper right',shadow=True,fancybox=True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
label_for_cabin_service = ['1.0','2.0','3.0','4.0','5.0']
data3 = airline_df['cabin_service'].value_counts().values
print(airline_df['cabin_service'].value_counts())
print(data3)
plt.figure(figsize=(4,4))
plt.title('Cabin service rating',bbox={'facecolor':'0.8', 'pad':5})
plt.pie(data3, labels = label_for_cabin_service,radius=1.6,autopct='%0.2f%%',shadow=True,textprops={'fontsize': 14})
plt.legend(loc='center',shadow=True,fancybox=True)
plt.show()

gp_by_cabin=airline_df.groupby('cabin')[['food_bev','entertainment']].mean().reset_index()
plt.rcParams['figure.figsize']=(4,4)
gp_by_cabin.plot(x="cabin", y=["food_bev", "entertainment"], kind="bar")

import matplotlib.pyplot as plt
import seaborn as sns
label_for_food_service = ['1.0','4.0','5.0','3.0','2.0']
data3 = airline_df['food_bev'].value_counts().values
plt.figure(figsize=(4,4))
plt.title('Food Beverage rating',bbox={'facecolor':'0.8', 'pad':5})
plt.pie(data3, labels = label_for_cabin_service,radius=1.6,autopct='%0.2f%%',shadow=True,textprops={'fontsize': 14})
plt.legend(loc='center',shadow=True,fancybox=True)
plt.show()

airline_df.dropna(subset=['recommended'],inplace=True)
plt.figure(figsize=(4,4))
sns.countplot(x=airline_df['seat_comfort'],hue=airline_df['recommended'])
plt.xlabel('seat_comfort')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

print(" ")
label_for_seat_service = ['1.0','4.0','3.0','5.0','2.0']
data4 = airline_df['seat_comfort'].value_counts().values
plt.figure(figsize=(4,4))
plt.title('Seat Comfort rating',bbox={'facecolor':'0.8', 'pad':5})
plt.pie(data4, labels = label_for_seat_service,radius=1.6,autopct='%0.2f%%',shadow=True,textprops={'fontsize': 14})
plt.legend(loc='center',shadow=True,fancybox=True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
airline_df.dropna(subset=['recommended'],inplace=True)
plt.figure(figsize=(4,4))
sns.countplot(x=airline_df['entertainment'],hue=airline_df['recommended'])
plt.xlabel('entertainment')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.show()

print(" ")

label_for_ent_service = ['1.0','4.0','5.0','3.0','2.0']
data5 = airline_df['entertainment'].value_counts().values
plt.figure(figsize=(4,4))
plt.title('Entertainment rating',bbox={'facecolor':'0.8', 'pad':5})
plt.pie(data5, labels = label_for_ent_service,radius=1.6,autopct='%0.2f%%',shadow=True,textprops={'fontsize': 14})
plt.legend(loc='center',shadow=True,fancybox=True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
airline_df.dropna(subset=['recommended'],inplace=True)
plt.figure(figsize=(4,4))
sns.countplot(x=airline_df['value_for_money'],hue=airline_df['recommended'])
plt.xlabel('value_for_money')
plt.ylabel('value_counts')
plt.xticks(rotation=90)
plt.show()

print(" ")

label_for_money = ['1.0','5.0','4.0','2.0','3.0']
data6 = airline_df['value_for_money'].value_counts().values
plt.figure(figsize=(4,4))
plt.title('value_for_money',bbox={'facecolor':'0.8', 'pad':5})
plt.pie(data6, labels = label_for_money,radius=1.6,autopct='%0.2f%%',shadow=True,textprops={'fontsize': 14})
plt.legend(loc='center',shadow=True,fancybox=True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
label_for_overall_rating = ['1.0','10.0','9.0','8.0','2.0','7.0','3.0','5.0','6.0','4.0']
data7 = airline_df['overall'].value_counts().values
print(airline_df['overall'].value_counts())
plt.figure(figsize=(4,4))
plt.title('overall rating',bbox={'facecolor':'0.8', 'pad':5})
plt.pie(data7, labels = label_for_overall_rating,radius=1.9,autopct='%0.2f%%',shadow=True,textprops={'fontsize': 14})
plt.legend(loc='center',shadow=True,fancybox=True)
plt.show()

airline_df = pd.read_csv("data_airline_reviews.csv")

if "recommended" not in airline_df.columns:
    print("Column 'recommended' is missing. Please check your dataset.")
else:
    print("Column 'recommended' found. Proceeding...")

numerical_columns = airline_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if "recommended" in numerical_columns:
    numerical_columns.remove("recommended")

# Detect and Remove Outliers using IQR
def remove_outliers_iqr(df, column):
    """Removes outliers from a specified column using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal to all numerical columns
for column in numerical_columns:
    airline_df = remove_outliers_iqr(airline_df, column)


X = airline_df.drop(columns=["recommended"])
y = airline_df["recommended"].map({'yes': 1, 'no': 0})  # Convert categorical target to numeric

print("\n Outliers removed. Cleaned dataset info:")
print(airline_df.info())


cleaned_filename = "cleaned_airline_reviews.csv"
airline_df.to_csv(cleaned_filename, index=False)
print(f"\n Cleaned dataset saved as '{cleaned_filename}'")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "cleaned_airline_reviews.csv"
airline_df = pd.read_csv(file_path)

numerical_columns = airline_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

plt.figure(figsize=(12, 12))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 1, i)
    sns.boxplot(data=airline_df[column], color="lightblue", width=0.3)
    plt.title(f"Boxplot for {column}", fontsize=12)
    plt.xlabel(column, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

X = airline_df.drop(columns=['recommended'])  # Drop target column
y = airline_df['recommended'].map({'yes': 1, 'no': 0})  # Convert to numeric

from sklearn.preprocessing import LabelEncoder

categorical_columns = X.select_dtypes(include=['object']).columns

label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X.fillna(0, inplace=True)

y = airline_df['recommended'].map({'yes': 1, 'no': 0})

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Shape of Features (X):", X.shape)
print("Shape of Target (y):", y.shape)
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

import matplotlib.pyplot as plt
import seaborn as sns
numeric_df = airline_df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(7, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_class = LogisticRegression(max_iter=2000, solver='saga')  # Increase iterations and use 'saga'
log_class.fit(X_train_scaled, y_train)


y_pred = log_class.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f" Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_logclass = log_class.predict(X_test_scaled)


confuse_mat_lr = confusion_matrix(y_test, y_pred_logclass)


plt.figure(figsize=(6, 4))
sns.heatmap(confuse_mat_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Recommended", "Recommended"],
            yticklabels=["Not Recommended", "Recommended"])
plt.title("Confusion Matrix of Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


y_pred_rf = rf_classifier.predict(X_test)


rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")


rf_report = classification_report(y_test, y_pred_rf)
print("\nClassification Report:\n")
print(rf_report)

rf_confusion_mat = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 6))
sns.heatmap(rf_confusion_mat, annot=True, fmt=".1f", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix of Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predict using scaled test set
y_pred_knn = knn_model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))


conf_mat = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Recommended", "Recommended"],
            yticklabels=["Not Recommended", "Recommended"])
plt.title("Confusion Matrix of KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




