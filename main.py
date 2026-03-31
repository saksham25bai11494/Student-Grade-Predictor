import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("student_grade_data.csv")

# Features and target
X = data[["StudyHours", "Attendance"]]
y = data["Grade"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# New student input
new_student = pd.DataFrame({
    "StudyHours": [5],
    "Attendance": [76]
})

# Predict grade
prediction = model.predict(new_student)

print("Predicted Grade:", prediction[0])
