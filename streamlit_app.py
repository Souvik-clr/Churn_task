import pickle
import streamlit as st
import pandas as pd
from PIL import Image
model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


def main():

	image = Image.open('images/icone.png')
	image2 = Image.open('images/image.png')
	st.image(image,use_column_width=False)
	add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?",
	("Online", "Batch"))
	st.sidebar.info('This app is created to predict Customer Churn')
	st.sidebar.image(image2)
	st.title("Predicting Customer Churn")
	if add_selectbox == 'Online':
		gender = st.selectbox('Gender:', ['male', 'female'])
		location= st.selectbox('location Option:', ['los_angeles', 'new_york', 'miami' ,'chicago', 'houston'])
		subscription_length_months = st.number_input('Number of months the customer has been with :', min_value=0, max_value=25, value=0)
		age= st.number_input('age :', min_value=0, max_value=71, value=0)
		monthly_bill= st.number_input('monthly_bill :', min_value=0, max_value=100, value=0)
		total_usage_gb= st.number_input('total_usage_gb :', min_value=0, max_value=500, value=0)
		output= ""
		output_prob = ""
		input_dict={
				"gender":gender ,
				"location":location,
				"subscription_length_months":subscription_length_months,
				"age":age,
				"monthly_bill":monthly_bill,
				"total_usage_gb":total_usage_gb
			}

		if st.button("Predict"):
			X = dv.transform([input_dict])
			y_pred = model.predict_proba(X)[0, 1]
			churn = y_pred >= 0.5
			output_prob = float(y_pred)
			output = bool(churn)
		st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))
	if add_selectbox == 'Batch':
		file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
		if file_upload is not None:
			data = pd.read_csv(file_upload)
			X = dv.transform([data])
			y_pred = model.predict_proba(X)[0, 1]
			churn = y_pred >= 0.5
			churn = bool(churn)
			st.write(churn)

if __name__ == '__main__':
	main()