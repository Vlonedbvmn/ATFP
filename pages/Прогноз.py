import streamlit as st
import pandas as pd
from neuralforecast.models import KAN, TimeLLM, TimesNet, NBEATSx, TimeMixer, PatchTST
from neuralforecast import NeuralForecast
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import io


# Initialize session state variables
if 'predicted' not in st.session_state:
    st.session_state.predicted = None

if 'predicted2' not in st.session_state:
    st.session_state.predicted2 = None

if 'plotp' not in st.session_state:
    st.session_state.plotp = None


# Set Streamlit page config
st.set_page_config(
    page_title="Прогноз",
    layout="wide",
    initial_sidebar_state="auto"
)

# Define button click functions

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def predd(datafre):
    try:
        print(datafre)
        qu = int(round(len(datafre)*0.1, 0))
        model = st.session_state.model_forecast
        print(2)
        preds = model.predict(df=datafre)
        print(preds)
        preds.rename(columns={'ds': st.session_state.date, str(st.session_state.inst_name): st.session_state.target}, inplace=True)
        preds.reset_index(drop=True, inplace=True)
        # preds = preds.drop(columns=['unique_id'], inplace=True)
        print(preds)
        print(-qu)
        pred1 = datafre[-qu:]
        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                     inplace=True)
        pred1.drop(columns=['unique_id'], inplace=True)
        print("1 df")
        print(pred1)
        print("2 df")
        print(preds)
        st.session_state.predicted = pd.concat([pred1, preds], ignore_index=True)
        print("finale df")
        print(len(st.session_state.predicted))
        print(st.session_state.predicted)
        st.session_state.predicted2 = preds
    except:
        st.warning('Щоб зробити прогноз оберіть спочатку модель)', icon="⚠️")

# Main function
if __name__ == "__main__":
    try:
        if st.session_state.inst_name is not None:
            with st.container():
                st.title("Прогноз")

            # Create two columns for buttons
            col1, col2 = st.columns(2)

            # Button for selecting experimental data
            with col1:
                st.markdown(f"### Модель: {st.session_state.inst_name}")
                dff = pd.DataFrame()
                dff["ds"] = st.session_state.df_forpred["ds"]
                dff["y"] = st.session_state.df_forpred["y"]
                dff["unique_id"] = st.session_state.df_forpred["unique_id"]
                print(dff)
                st.button(label="Прогноз", key="pr", on_click=predd,
                          args=(dff,))

            # Button for selecting own data
            with col2:
                st.markdown(f"### Дані:")
                with st.expander("Подивитися обраний датасет:"):
                    st.write(st.session_state.df)



            st.divider()
            st.markdown(f"### Результати прогнозу")
            if st.session_state.predicted2 is not None:
                col3, col4 = st.columns(2)
                with col3:
                    with st.expander("Подивитись прогнозні значення:"):
                        st.write(st.session_state.predicted2)
                with col4:

                    st.download_button(
                        label="Завантажити прогноз як файл .csv",
                        data=st.session_state.predicted2.to_csv().encode("utf-8"),
                        file_name="prediction.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        label="Завантажити прогноз як файл .xlsx",
                        data=to_excel(st.session_state.predicted2),
                        file_name="prediction.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                st.divider()

                st.markdown(f"### Графік прогнозу")

                last_days = st.session_state.predicted.tail(st.session_state.horiz)
                rest_of_data = st.session_state.predicted.iloc[:-st.session_state.horiz]

                # Create the plotly figure
                st.session_state.plotp = go.Figure()

                # Plot the data except the last seven days
                st.session_state.plotp.add_trace(go.Scatter(
                    x=rest_of_data[st.session_state.date],
                    y=rest_of_data[st.session_state.target],
                    mode='lines',
                    name='Дані',
                    line=dict(color='blue')
                ))

                # Plot the last seven days in a different color
                st.session_state.plotp.add_trace(go.Scatter(
                    x=last_days[st.session_state.date],
                    y=last_days[st.session_state.target],
                    mode='lines',
                    name='Прогноз',
                    line=dict(color='red')
                ))

                # Update layout (optional)
                st.session_state.plotp.update_layout(
                    xaxis_title='Дата',
                    yaxis_title='Значення'
                )

                # Show the plot
                st.plotly_chart(st.session_state.plotp, use_container_width=True)
        else:
            st.warning('Щоб зробити прогноз оберіть спочатку модель', icon="⚠️")
    except:
        st.warning('Щоб зробити прогноз оберіть спочатку модель', icon="⚠️")