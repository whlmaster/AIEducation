import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# %matplotlib inline

st.title('Оценка результативности рекламных акций')

st.text("Клиентам Банка рассылаются рекламные предложения, некоторые на них откликаются")

# импорт данных
from eda import tab_analize

st.header("Кто наш клиент и кто откликается на рекламные предложения")

st.dataframe(pd.DataFrame([tab_analize.median(numeric_only=True),
                           tab_analize.query('target == 1').median(numeric_only=True)]).iloc[:, 1:].T)

st.write(
    "На основании выборки медианный клиент, отликвнувшийся на маркетинговое предложение - мужчина в "
    "возврасте 36 лет, с доходом 14 тыс. у.е. в месяц, имеющий одного ребенка и одного члена семьи на "
    "иждивении, имеет работу и один действующий кредит"
)

st.header("Половозрастная структура клиентов")
#check = st.selectbox('Выбор клиентов по отклику',
#                     ("С откликами", "Без откликов", "Все клиенты"))

#check_dict={"С откликами":1, "Без откликов":0, "Все клиенты":5}

from eda import tab_values
g = (tab_values#.loc[tab_values.target.eq(check_dict.get(check))]
     .pivot_table(values='target', index=['gender', 'age'], aggfunc='mean')
     .reset_index())
g.loc[g.gender.eq(1), 'target'] = g.target.mul(-1)

fig = plt.subplots(figsize=(7, 6))
population = sns.barplot(data=g, x='target', y='age',
            hue='gender', orient='horizontal',
            dodge=False, palette='Set1')
fig = population.figure
st.pyplot(fig)

st.header("Графики распределения откликов в зависимости от признака")
selector = st.selectbox(
    'Выберите признак',
    (tab_analize.columns[2:-4]))

bin_pointer = st.slider('Количество групп',
                        min_value=2,
                        max_value=10,
                        value=6,
                        step=1
                        )

fig = plt.figure(figsize=(10, 4))
sns.histplot(x=tab_analize[selector],
             hue='target',
             data=tab_analize,
             multiple="dodge",
             bins=bin_pointer,
             kde=True,
             stat="probability",
             common_norm=False,
             palette='Set1',
             shrink=0.6).set(title='Доля откликов по группам')
st.pyplot(fig)

st.header('Матрица корреляций')
st.write('Сильной линейной зависимости между откликом и факторами не выявлено.'
         'Признаки кол-во ссуд \ количество закрытых ссуд, наличие работы \ пенсионер -'
         'попарно мультиколлениарны.'
         'Статус пенсионера\возрат и кол-во детей\кол-во иждивенцев имеют умеренную линейную зависимость.')

fig, ax = plt.subplots()
sns.set(font_scale=0.8)
sns.heatmap(
    tab_values.corr(),
    cmap='RdBu_r',
    annot=True,
    fmt=".2f",
    vmin=-1,
    vmax=1,
    ax=ax)
st.pyplot(fig)


X_train, X_test, y_train, y_test = train_test_split(tab_values.drop('target', axis=1),
                                                    tab_values['target'],
                                                    test_size=0.20,
                                                    random_state=42)

ss = StandardScaler()
ss.fit(X_train)
X_train = pd.DataFrame(ss.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(ss.transform(X_test), columns=X_test.columns)

model = LogisticRegression()
model.fit(X_train, y_train)

st.header("Предсказание вероятности отклика")
bin_pointer = st.slider('Порог чувствительности',
                        min_value=0.0,
                        max_value=1.0,
                        value=0.66,
                        step=0.01
                        )
probs_churn = model.predict_proba(X_test)[:,1]
classes = probs_churn < bin_pointer

st.dataframe(pd.DataFrame([['Доля верных ответов', 'Точность', 'Полнота', 'F1-мера'],
                           [accuracy_score(y_test, classes), precision_score(y_test, classes), recall_score(y_test, classes), f1_score(y_test, classes)]
                           ]
                          ).T
             )

with st.form("Введите данные клиента для предсказания:"):
    age = st.number_input('Возраст клиента, лет:', min_value=1, max_value=100, value=35, step=1)
    gender = st.selectbox("Пол клиента:", ['Мужской', 'Женский'])
    income = st.number_input('Личный доход, у.е.:', min_value=0, value=14000)
    child = st.number_input('Количество детей:', min_value=0, max_value=100, value=0, step=1)
    dep = st.number_input('Количество иждивенцев:', min_value=0, max_value=100, value=0, step=1)
    cred_active = st.number_input('Количество действующих кредитов:', min_value=0, max_value=100, value=0, step=1)
    cred_hist = st.number_input('Количество закрытых кредитов:', min_value=0, max_value=100, value=2, step=1)
    job = sum([st.checkbox('Наличие работы')])
    pens = sum([st.checkbox('Пенсионер')])

    submitted = st.form_submit_button("Получить результат")

    translatetion = {
        "Мужской": 1,
        "Женский": 0
    }

    data = {
        "loans_count": cred_active,
        "closed_fl": cred_hist,
        "personal_income": income,
        "age": age,
        "socstatus_work_fl": job,
        "socstatus_pens_fl": pens,
        "gender": translatetion[gender],
        "child_total": child,
        "dependants": dep
    }

    df = pd.DataFrame(data, index=[0])

    if submitted:
       X_request=pd.DataFrame(ss.transform(df),
                              columns=X_train.columns)
       st.write(f'Вероятность отклика: {model.predict_proba(X_request)[0][1]:.2%}')
