import numpy as np
import joblib
model=joblib.load('model_heartdisease.pkl')

import streamit as st
sex=('M','F')
chestPainType=('ASY','NAP','ATA','TA')
restingECG=('Normal','LVH','ST')
st_Slope=('Flat','Up','Down')
def main():
    st.title('Heart Disease Prediction')
    
    Age=st.text_input('Enter Your Age')
    Sex=st.selectbox('Sex',sex)
    ChestPainType=st.selectbox('ChestPainType',chestPainType)
    RestingBP=st.text_input('Enter RestingBP')
    Cholesterol=st.text_input('Enter Cholesterol level')
    FastingBS=st.text_input('Enter FastingBS')
    RestingECG=st.selectbox('RestingECG',restingECG)
    MaxHR=st.text_input('Enter MaxHR')
    Oldpeak=st.text_input('Enter Oldpeak')
    ST_Slope=st.selectbox('ST_Slope',st_Slope)
    
    ok=st.button('Predict Disease')
    if ok:
        X=np.array([[Age,Sex, ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,
                    MaxHR, Oldpeak,ST_Slope]])
        prep=model['preprocess'].transform(X)
        y_pred=model['predict'].predict(prep)
        st.subheader(f'The Patient has {y_pred}')
if __name__ == "__main__":
    main()