import streamlit as st
import pandas as pd
from neuralforecast.models import KAN, TimeLLM, TimesNet, NBEATSx, TimeMixer, PatchTST
from neuralforecast import NeuralForecast
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import numpy as np
import io
import os
import random
import time
from groq import Groq

means = {"Місяць": "M",
         "Година": "h",
         "Рік": "Y",
         "Хвилина": "T",
         "Секунда": "S",
         "День": "D",
         }

client = Groq(api_key="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")



if "messages1" not in st.session_state:
    st.session_state.messages1 = [{"role": "user", "content": "Здійсни прогнозування на наступні 7 днів"},
                                  {"role": "user", "content": "Здійсни тестування датасету на аномалії"},
                                  {"role": "user", "content": "Здійсни прогнозування на 2 тижні за допомогою моделі TimeMixer"}]

if "no_d" not in st.session_state:
    st.session_state.no_d = None 


if "messages" not in st.session_state:
    st.session_state.messages = []

if "fig_b" not in st.session_state:
    st.session_state.fig_b = None
if "dataai" not in st.session_state:
    st.session_state.dataai = None
if "m1" not in st.session_state:
    st.session_state.m1 = None
if "m2" not in st.session_state:
    st.session_state.m2 = None

def response_1(chr):
    response = chr
    for word in response.split():
                yield word + " "
                time.sleep(0.1)

# Streamed response emulator
def response_generator(datafra, res):
    st.session_state.fig_b = None
    st.session_state.dataai = None
    if st.session_state.date_not_n:
        start_date = pd.to_datetime('2024-01-01')
        datafra['ds'] = start_date + pd.to_timedelta(datafra['ds'] - 1, means[st.session_state.freq])

    datafra['ds'] = pd.to_datetime(datafra['ds'])
    datafra = datafra.drop_duplicates(subset=['ds'])
    datafra = datafra.set_index('ds').asfreq(means[st.session_state.freq])
    datafra = datafra.reset_index()
    datafra['y'] = datafra['y'].interpolate()
    datafra["unique_id"] = [0 for i in range(1, len(datafra) + 1)]
    print("s;kgfoshdisdifsdf")
    print(datafra)

    chatco = client.chat.completions.create(
        messages=[
            {
               "role": "user",
                "content": f"З цього тексту '{res}' знайди потрібні ключові слова та видай мені відповідь за таким шаблоном:'model:  horizon:  input_size:  task:  '."
                           f"model може бути із тільки цього списку: [NBEATSx, KAN, PatchTST, TimeMixer, TimesNet, Авто-вибір, None]; task: [Anomaly, Forecasting, None]; а horizon пиши лише значення без пояснення частоти.  В разі якщо нема якогось компоненту на його місці пиши 'None'. Якщо нема жодного пиши тільки 'уточніть запит'. При наданні відповіді стого слідуй інструкції, тобто відповідай строго за шаблоном. Не пиши зайвого тексту!!! Не пиши коми!!!",
            }
        ],
        model="llama-3.1-70b-versatile"
    )
    respo = chatco.choices[0].message.content
    response = "Уточніть, будь ласка, Ваш запит"
    try:
        mdl = respo.split()[1]
        hrz = respo.split()[3]
        inp_sz = respo.split()[5]
        tsk = respo.split()[7]

        q = int(round(len(datafra) * 0.01, 0))

        if tsk == "Forecasting":
            st.session_state.m1 = "Ось табличка з данми Вашого прогнозу"
            st.session_state.m2 = "Та також графік Вашого прогнозу. Синім зображені дані до, а червоним зображено сам прогноз"
            response = "Дякую за Ваш запит, ось результати прогнозування"
            if mdl == "KAN":
                if hrz == "None":
                    response = "Уточніть на скільки вперед робити прогноз"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                KAN(h=int(hrz),
                                    input_size=int(hrz)*q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "KAN": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"KAN": st.session_state.target}).drop(["unique_id"], axis=1)
                        # Show the plot
                    else:
                        fcst = NeuralForecast(
                            models=[
                                KAN(h=int(hrz),
                                    input_size=int(inp_sz),
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "KAN": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"KAN": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "NBEATSx":
                if hrz == "None":
                    response = "Уточніть на скільки вперед робити прогноз"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                NBEATSx(h=int(hrz),
                                    input_size=int(hrz)*q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "NBEATSx": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
                        # Show the plot
                    else:
                        fcst = NeuralForecast(
                            models=[
                                NBEATSx(h=int(hrz),
                                    input_size=int(inp_sz),
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "NBEATSx": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "TimesNet":
                if hrz == "None":
                    response = "Уточніть на скільки вперед робити прогноз"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                TimesNet(h=int(hrz),
                                    input_size=int(hrz)*q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimesNet": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"TimesNet": st.session_state.target}).drop(["unique_id"], axis=1)
                        # Show the plot
                    else:
                        fcst = NeuralForecast(
                            models=[
                                TimesNet(h=int(hrz),
                                    input_size=int(inp_sz),
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimesNet": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"TimesNet": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "TimeMixer":
                if hrz == "None":
                    response = "Уточніть на скільки вперед робити прогноз"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                TimeMixer(h=int(hrz),
                                    input_size=int(hrz)*q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    # start_padding_enabled=True,
                                    n_series=1
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimeMixer": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"TimeMixer": st.session_state.target}).drop(["unique_id"], axis=1)
                        # Show the plot
                    else:
                        fcst = NeuralForecast(
                            models=[
                                TimeMixer(h=int(hrz),
                                    input_size=int(inp_sz),
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    # start_padding_enabled=True,
                                    n_series=1
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimeMixer": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"TimeMixer": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "PatchTST":
                if hrz == "None":
                    response = "Уточніть на скільки вперед робити прогноз"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                PatchTST(h=int(hrz),
                                    input_size=int(hrz)*q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "PatchTST": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"PatchTST": st.session_state.target}).drop(["unique_id"], axis=1)
                        # Show the plot
                    else:
                        fcst = NeuralForecast(
                            models=[
                                PatchTST(h=int(hrz),
                                    input_size=int(inp_sz),
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "PatchTST": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"PatchTST": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "Авто-вибір":
                if hrz == "None":
                    response = "Уточніть на скільки вперед робити прогноз"
                else:
                    fcst = NeuralForecast(
                        models=[
                            KAN(h=int(hrz),
                                input_size=int(hrz) * q,
                                # output_size=horizon,
                                max_steps=30,
                                scaler_type='standard',
                                start_padding_enabled=True
                                ),
                            TimesNet(h=int(hrz),
                                     input_size=int(hrz) * q,
                                     # output_size=horizon,
                                     max_steps=30,
                                     scaler_type='standard',
                                     start_padding_enabled=True
                                     ),
                            TimeMixer(h=int(hrz),
                                      input_size=int(hrz) * q,
                                      # output_size=horizon,
                                      max_steps=30,
                                      scaler_type='standard',
                                      # start_padding_enabled=True,
                                      n_series=1
                                      ),
                            PatchTST(h=int(hrz),
                                     input_size=int(hrz) * q,
                                     # output_size=horizon,
                                     max_steps=30,
                                     scaler_type='standard',
                                     start_padding_enabled=True
                                     ),
                            NBEATSx(h=int(hrz),
                                    input_size=int(hrz)* q,
                                    # output_size=horizon,
                                    max_steps=30,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),

                        ],
                        freq=means[st.session_state.freq]
                    )

                    Y_train_df = datafra[:-int(hrz)]
                    Y_test_df = datafra[-int(hrz):]
                    fcst.fit(df=Y_train_df)
                    forecasts = fcst.predict(futr_df=Y_test_df)
                    print(forecasts)
                    results = {}
                    for i in ["KAN", "TimesNet", "TimeMixer", "PatchTST", "NBEATSx"]:
                        results[i] = mean_squared_error(Y_test_df["y"], forecasts[i])

                    key_with_min_value = min(results, key=results.get)

                    if key_with_min_value == "KAN":
                        fcst = NeuralForecast(
                            models=[
                                KAN(h=int(hrz),
                                    input_size=int(hrz)* q,
                                    # output_size=horizon,
                                    max_steps=50,
                                    scaler_type='standard',
                                    start_padding_enabled=True
                                    ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "KAN": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"KAN": st.session_state.target}).drop(["unique_id"], axis=1)
                    if key_with_min_value == "NBEATSx":
                        fcst = NeuralForecast(
                            models=[
                                NBEATSx(h=int(hrz),
                                         input_size=int(hrz)* q,
                                         # output_size=horizon,
                                         max_steps=50,
                                         scaler_type='standard',
                                         start_padding_enabled=True
                                         ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "NBEATSx": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
                    if key_with_min_value == "PatchTST":
                        fcst = NeuralForecast(
                            models=[
                                PatchTST(h=int(hrz),
                                         input_size=int(hrz)* q,
                                         # output_size=horizon,
                                         max_steps=50,
                                         scaler_type='standard',
                                         start_padding_enabled=True
                                         ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "PatchTST": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"PatchTST": st.session_state.target}).drop(["unique_id"], axis=1)
                    if key_with_min_value == "TimesNet":
                        fcst = NeuralForecast(
                            models=[
                                TimesNet(h=int(hrz),
                                         input_size=int(hrz)* q,
                                         # output_size=horizon,
                                         max_steps=50,
                                         scaler_type='standard',
                                         start_padding_enabled=True
                                         ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimesNet": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"TimesNet": st.session_state.target}).drop(["unique_id"], axis=1)
                    if key_with_min_value == "TimeMixer":
                        fcst = NeuralForecast(
                            models=[
                                TimeMixer(h=int(hrz),
                                         input_size=int(hrz)* q,
                                         # output_size=horizon,
                                         max_steps=50,
                                         scaler_type='standard',
                                         # start_padding_enabled=True,
                                         n_series=1
                                         ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "TimeMixer": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"TimeMixer": st.session_state.target}).drop(["unique_id"], axis=1)
            if mdl == "None":
                if hrz == "None":
                    response = "Уточніть на скільки вперед робити прогноз"
                else:
                    if inp_sz == "None":
                        fcst = NeuralForecast(
                            models=[
                                NBEATSx(h=int(hrz),
                                        input_size=int(hrz) * q,
                                        # output_size=horizon,
                                        max_steps=50,
                                        scaler_type='standard',
                                        start_padding_enabled=True
                                        ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "NBEATSx": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
                        # Show the plot
                    else:
                        fcst = NeuralForecast(
                            models=[
                                NBEATSx(h=int(hrz),
                                        input_size=int(inp_sz),
                                        # output_size=horizon,
                                        max_steps=50,
                                        scaler_type='standard',
                                        start_padding_enabled=True
                                        ),
                            ],
                            freq=means[st.session_state.freq]
                        )
                        fcst.fit(df=datafra)
                        qu = int(round(len(datafra) * 0.1, 0))
                        print(2)
                        preds = fcst.predict(df=datafra)
                        print(preds)
                        preds.rename(
                            columns={'ds': st.session_state.date, "NBEATSx": st.session_state.target},
                            inplace=True)
                        preds.reset_index(drop=True, inplace=True)
                        # preds = preds.drop(columns=['unique_id'], inplace=True)
                        print(preds)
                        print(-qu)
                        pred1 = datafra[-qu:]
                        pred1.rename(columns={'ds': st.session_state.date, "y": st.session_state.target},
                                     inplace=True)
                        pred1.drop(columns=['unique_id'], inplace=True)
                        print("1 df")
                        print(pred1)
                        print("2 df")
                        print(preds)
                        predis = pd.concat([pred1, preds], ignore_index=True)
                        print("finale df")

                        pred2 = preds
                        last_days = predis.tail(int(hrz))
                        rest_of_data = predis.iloc[:-int(hrz)]
                        print(last_days)

                        # Create the plotly figure
                        chr = go.Figure()

                        # Plot the data except the last seven days
                        chr.add_trace(go.Scatter(
                            x=rest_of_data[st.session_state.date],
                            y=rest_of_data[st.session_state.target],
                            mode='lines',
                            name='Дані',
                            line=dict(color='blue')
                        ))

                        # Plot the last seven days in a different color
                        chr.add_trace(go.Scatter(
                            x=last_days[st.session_state.date],
                            y=last_days[st.session_state.target],
                            mode='lines',
                            name='Прогноз',
                            line=dict(color='red')
                        ))

                        # Update layout (optional)
                        chr.update_layout(
                            xaxis_title='Дата',
                            yaxis_title='Значення'
                        )
                        st.session_state.fig_b = chr
                        st.session_state.dataai = pred2.rename.rename(columns={"NBEATSx": st.session_state.target}).drop(["unique_id"], axis=1)
        elif tsk == "Anomaly":
            st.session_state.m1 = "Ось табличка з даними після проведення тестування на аномалії"
            st.session_state.m2 = "Та також графік Вашого прогнозу. Синім зображені Ваші дані, зеленим - прогнозовані а червоним крапками місця з аномаліями"
            response = "Дякую за Ваш запит, ось резульати проведення тестування на аномалії"
            model = NeuralForecast(
                models=[
                    NBEATSx(h=len(datafra),
                            input_size=30 * q,
                            # output_size=horizon,
                            max_steps=100,
                            # scaler_type='standard',
                            start_padding_enabled=True
                            ),

                ],
                freq=means[st.session_state.freq]
            )
            model.fit(datafra)  # Use the entire dataset for training

            # Generate predictions
            predictions = model.predict(datafra.head(1))
            print(predictions)
            datafra['NBEATSx'] = predictions['NBEATSx']
            datafra['residuals'] = np.abs(datafra['y'] - datafra['NBEATSx'])

            # Set anomaly threshold (adjust based on domain knowledge)
            threshold = 4 * datafra['residuals'].std()
            datafra['anomaly'] = datafra['residuals'] > threshold

            # Plot actual, predicted values, and anomalies using plotly
            fig = go.Figure()

            # Add actual values
            fig.add_trace(go.Scatter(x=datafra['ds'], y=datafra['y'], mode='lines', name='Дані', line=dict(color='blue')))

            # Add predicted values
            fig.add_trace(go.Scatter(x=datafra['ds'], y=datafra['NBEATSx'], mode='lines', name='Прогнозовано',
                                     line=dict(color='green')))

            # Highlight anomalies
            anomalies = datafra[datafra['anomaly'] == True]
            fig.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], mode='markers', name='Аномалія',
                                     marker=dict(color='red', size=8)))

            # Add title and labels
            fig.update_layout(
                title='Графік аномалій',
                xaxis_title='Дата',
                yaxis_title='Значення',
                template='plotly_white'
            )

            datafra = datafra.rename(columns={"NBEATSx": "preds"})
            # Show the plot
            st.session_state.fig_b = fig
            st.session_state.dataai = datafra.drop(['unique_id', 'residuals'], axis=1)
        elif tsk == "None":
            response = "Вибачте, але здається що Ви не вказали, що конкретно хочете робити"

    except: pass
    for word in response.split():
                yield word + " "
                time.sleep(0.1)




st.set_page_config(
    page_title="Аналіз аномалій",
    layout="wide",
    initial_sidebar_state="auto"
)




if __name__ == "__main__":
    if st.session_state.df is not None:
        st.session_state.no_d = None 
        print(st.session_state.fig_b)
        print(st.session_state.dataai)
        ds_for_pred = pd.DataFrame()
        ds_for_pred["y"] = st.session_state.df[st.session_state.target]
        try:
            ds_for_pred["ds"] = st.session_state.df[st.session_state.date]
            st.session_state.date_not_n = False
        except:
            st.session_state.date_not_n = True
            ds_for_pred['ds'] = [i for i in range(1, len(ds_for_pred) + 1)]
        st.title("ШІ помічник")
        st.write(" ")
        st.markdown("## Приклади запитів до ШІ помічника:")
        for message in st.session_state.messages1:
            with st.chat_message(message["role"]):
                # if isinstance(message["content"],str):
                #     st.markdown(message["content"])
                # elif isinstance(message["content"],st.delta_generator.DeltaGenerator):
                #     st.plotly_chart(message["content"])
                # else:
                st.write(message["content"])
        st.write(" ")
        st.markdown("## Чат")
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # if isinstance(message["content"],str):
                #     st.markdown(message["content"])
                # elif isinstance(message["content"],st.delta_generator.DeltaGenerator):
                #     st.plotly_chart(message["content"])
                # else:
                st.write(message["content"])
        with st.chat_message("assistant"):
            st.write_stream(response_1(f"Зараз працюю з обраним Вами набором даних: {st.session_state.name}"))
        # Accept user input
        if prompt := st.chat_input("Напишіть свій запит і отримайте відповідь"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # st.session_state.messages.append({"role": "assistant", "content": "Дякую за запитання, інтерпритую ваш запит до моделі прогнозування"})

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                gen = response_generator(ds_for_pred, prompt)
                response = st.write_stream(gen)
                st.session_state.messages.append({"role": "assistant", "content": response})
                if st.session_state.dataai is not None:
                    r1 = st.write_stream(response_1(st.session_state.m1))
                    dai = st.write(st.session_state.dataai)
                    r2 = st.write_stream(response_1(st.session_state.m2))
                    chart = st.plotly_chart(st.session_state.fig_b, use_container_width=True)
                    print(dai)
                    print("-"*1000)
                    st.session_state.messages.append({"role": "assistant", "content": r1})
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.dataai})
                    st.session_state.messages.append({"role": "assistant", "content": r2})
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.fig_b})



    else:
        st.session_state.no_d = True
        st.title("ШІ помічник")
        st.write(" ")
        st.markdown("## Приклади запитів до ШІ помічника:")
        for message in st.session_state.messages1:
            with st.chat_message(message["role"]):
                # if isinstance(message["content"],str):
                #     st.markdown(message["content"])
                # elif isinstance(message["content"],st.delta_generator.DeltaGenerator):
                #     st.plotly_chart(message["content"])
                # else:
                st.write(message["content"])
        st.write(" ")
        st.markdown("## Чат")
        with st.chat_message("assistant"):
            st.write_stream(response_1("Перед тим як працювати зі мною, оберіть дані з якими Ви будете працювати у розділі 'Дані'"))
