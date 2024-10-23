import streamlit as st
import pandas as pd
from neuralforecast.models import KAN, TimeLLM, TimesNet, NBEATSx, TimeMixer, PatchTST
from statsforecast.models import MSTL
from neuralforecast import NeuralForecast
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import numpy as np
import io

# st.set_page_config(
#     page_title="Аналіз аномалій",
#     layout="wide",
#     initial_sidebar_state="auto"
# )


means = {"Місяць": "M",
         "Година": "h",
         "Рік": "Y",
         "Хвилина": "T",
         "Секунда": "S",
         "День": "D",
         }

if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'fig_a' not in st.session_state:
    st.session_state.fig_a = None
if 'mse' not in st.session_state:
    st.session_state.mse = None
if 'inst_name' not in st.session_state:
    st.session_state.inst_name = None
if 'model_forecast' not in st.session_state:
    st.session_state.model_forecast = None
if 'df_forpred' not in st.session_state:
    st.session_state.df_forpred = None
if 'horiz' not in st.session_state:
    st.session_state.horiz = None
if 'date_not_n' not in st.session_state:
    st.session_state.date_not_n = None
if 'datanom' not in st.session_state:
    st.session_state.datanom = None

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

@st.cache_data(show_spinner="Проводимо тестування...")
def anomal(datafra, freqs):
    if st.session_state.date_not_n:
        start_date = pd.to_datetime('2024-01-01')
        freqs = "D"
        datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, freqs)

    datafra['ds'] = pd.to_datetime(datafra['ds'])
    datafra = datafra.drop_duplicates(subset=['ds'])
    datafra = datafra.set_index('ds').asfreq(freqs)
    datafra = datafra.reset_index()
    datafra['y'] = datafra['y'].interpolate()
    datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
    print("s;kgfoshdisdifsdf")
    print(datafra)

    q = int(round(len(datafra) * 0.01, 0))

    # fcst = NeuralForecast(
    #     models=[
    #         NBEATSx(h=len(datafra),
    #                 input_size=14*q,
    #                 # output_size=horizon,
    #                 max_steps=20,
    #                 scaler_type='standard',
    #                 start_padding_enabled=True
    #                 ),
    #
    #     ],
    #     freq=freqs
    # )


    # Define and train the NBEATSx model
    model = NeuralForecast(
        models=[
            NBEATSx(h=len(datafra),
                    input_size=30 * q,
                    # output_size=horizon,
                    max_steps=100,
                    scaler_type='standard',
                    start_padding_enabled=True
                    ),

        ],
        freq=freqs
    )
    model.fit(datafra)  # Use the entire dataset for training

    # Generate predictions
    predictions = model.predict(datafra.head(1))
    print("preds", "-"*100)
    print(predictions)
    datafra['NBEATSx'] = predictions['NBEATSx']
    datafra['residuals'] = np.abs(datafra['y'] - datafra['NBEATSx'])

    # Set anomaly threshold (adjust based on domain knowledge)
    threshold = 4 * datafra['residuals'].std()
    datafra['anomaly'] = datafra['residuals'] > threshold

    # # Plot actual, predicted values, and anomalies using plotly

    st.session_state.datanom = datafra.drop(['unique_id', 'residuals'], axis=1)
    if st.session_state.date_not_n == True:
        st.session_state.datanom["ds"] = [i for i in range(1, len(st.session_state.datanom) + 1)]
    print("preds", "-" * 100)
    print(st.session_state.datanom)


# if __name__ == "__main__":

if st.session_state.df is not None:
    # st.session_state.datanom = None
    ds_for_pred = pd.DataFrame()
    ds_for_pred["y"] = st.session_state.df[st.session_state.target]
    try:
        ds_for_pred["ds"] = st.session_state.df[st.session_state.date]
        st.session_state.date_not_n = False
        ds_for_pred['ds'] = pd.to_datetime(ds_for_pred['ds'])
    except:
        st.session_state.date_not_n = True
        ds_for_pred['ds'] = [i for i in range(1, len(ds_for_pred) + 1)]

    # if st.session_state.date_not_n:
    #     start_date = pd.to_datetime('2024-01-01')
    #     ds_for_pred['ds'] = start_date + pd.to_timedelta(ds_for_pred['ds'] - 1, rarety)

    # ds_for_pred['ds'] = pd.to_datetime(ds_for_pred['ds'])
    # ds_for_pred = ds_for_pred.set_index('ds').asfreq(rarety)
    # ds_for_pred = ds_for_pred.reset_index()
    # ds_for_pred['y'] = ds_for_pred['y'].interpolate()
    # ds_for_pred["unique_id"] = [0 for i in range(1, len(ds_for_pred) + 1)]
    # print("s;kgfoshdisdifsdf")
    # print(ds_for_pred)
    print(ds_for_pred)
    # st.session_state.df_forpred = ds_for_pred

    with st.container():
        st.title("Тестування часового ряду на аномалії")
        # st.markdown("### ")
        st.markdown("#### Тестування часових рядів на виявлення аномалій є важливим етапом аналізу, що дозволяє ідентифікувати нехарактерні або несподівані зміни в даних, які можуть свідчити про значущі події або проблеми у функціонуванні системи. Аномалії можуть бути ознакою технічних несправностей, системних збоїв або навіть випадків шахрайства у фінансових даних. Вчасне виявлення таких відхилень сприяє запобіганню критичним помилкам та мінімізації ризиків.")

    st.markdown("### ")
    # fr = st.selectbox("Оберіть частоту запису даних в ряді:",
    #                   ["Місяць", "День", "Рік", "Хвилина", "Секунда", "Година"])
    try:

        st.button(label="Провести тестування", key="kan", on_click=anomal,
                  args=(ds_for_pred, means[st.session_state.freq]))

    except:
        st.warning('Надано не коректні гіперпараметри', icon="⚠️")

    st.divider()

    if st.session_state.datanom is not None:
        st.markdown("# Результат проведення тестування")
        datafra = st.session_state.datanom.rename(columns={"NBEATSx": "preds"})
        print("preds", "-" * 100)
        print(datafra)
        col3, col4 = st.columns(2)
        with col3:
            with st.expander("Подивитись тест даних на аномалії:"):
                st.write(st.session_state.datanom)
        with col4:
            st.download_button(
                label="Завантажити тест як файл .csv",
                data=st.session_state.datanom.to_csv().encode("utf-8"),
                file_name="anomaly.csv",
                mime="text/csv"
            )
            st.download_button(
                label="Завантажити тест як файл .xlsx",
                data=to_excel(st.session_state.datanom),
                file_name="anomaly.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        st.divider()
        sl = st.select_slider(
            "Оберіть горизонт даних:",
            options=[i for i in range(1, len(datafra))]
        )
        fig = go.Figure()

        # Add actual values
        fig.add_trace(
            go.Scatter(x=datafra[:sl]['ds'], y=datafra[:sl]['y'], mode='lines', name='Дані', line=dict(color='blue')))

        # Add predicted values
        print("jkghhdfgihdfiopjajkiopdfjlkbjklopkdjklfbjkaopwkwdjkbn jkiopaibhn dhjfiopijiahbe rjiofvh adjiofvh jiobhfv ghiojHVJ DFHJIObhwv hdiou0fohvgHIOJOHBJVFHIOUPshGCDFUIOSVJGHDVUIFPHGVGCAGHUSIDHFGVGGUIASDGVGFUIOPAHVGUIF8HVAGUIDDCFHUIOSDGHVGFUOAHGSHVDFUIOHGV")
        print(datafra[:sl]['preds'])
        fig.add_trace(
            go.Scatter(x=datafra[:sl]['ds'], y=datafra[:sl]['preds'], mode='lines', name='Прогнозовано', line=dict(color='green')))

        # Highlight anomalies
        anomalies = datafra[:sl][datafra['anomaly'] == True]
        fig.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], mode='markers', name='Аномалія',
                                 marker=dict(color='red', size=8)))

        # Add title and labels
        fig.update_layout(
            title='Графік аномалій',
            xaxis_title='Дата',
            yaxis_title='Значення',
            template='plotly_white'
        )

        # Show the plot
        st.session_state.fig_a = fig
        co1, co2 = st.columns([1, 4])
        with co2:
            st.plotly_chart(st.session_state.fig_a, use_container_width=True)
        anomalie_count = 0
        for i in datafra[:sl]['anomaly'].values.tolist():
            if i:
                anomalie_count += 1
        with co1:
            st.markdown("# ")
            st.metric(label="К-ть аномалій", value=anomalie_count)

    # st.button("Train", type="primary", on_click=save_performance, args=((model, k)))
    #
    # with st.expander("See full dataset"):
    #     st.write(wine_df)
    #
    # if len(st.session_state['score']) != 0:
    #     st.subheader(f"The model has an F1-Score of: {st.session_state['score'][-1]}")
else:
    st.warning('Для проведення тесту на аномалії, оберіть дані', icon="⚠️")