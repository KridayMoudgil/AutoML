from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Head Content 
st.title('AutoML')
st.write('We help you create Machine Learning Models without writing any piece of code.')

# File Upload
dataset = st.file_uploader("Choose a CSV File", type='.csv')

if dataset is not None:
    df = pd.read_csv(dataset)
    st.write(df.head())

    target = st.selectbox('Column to be predicted: ', df.columns)
    use_columns = st.multiselect('Column(s) to be used: ', df.columns)

    if st.button('Train Model'):
        X = df[use_columns]
        y = df[[target]]

        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=4)

        model = LinearRegression().fit(train_x, train_y)
        predictions = model.predict(test_x)
        acc = r2_score(test_y, predictions)

        # Plotting only for 1D regression
        if len(use_columns) == 1:
            x_vals = train_x.squeeze()
            y_vals = train_y.squeeze()
            predicted_line = model.coef_[0][0] * x_vals + model.intercept_[0]
            fig, ax = plt.subplots()
            ax.scatter(x_vals, y_vals, color='blue', label='Actual Data')
            ax.plot(x_vals, predicted_line, color='red', label='Fit Line')
            ax.set_xlabel(use_columns[0])
            ax.set_ylabel(target)
            ax.legend()
            st.pyplot(fig)

        st.write("Model's R2 Score: ", np.round(acc, 2))
        if acc >= 0.8:
            st.success("✅ High Accuracy: Model predictions are stable and accurate.")
        else:
            st.warning("⚠️ Accuracy could be improved: Try tweaking features or cleaning data.")

        st.balloons()
