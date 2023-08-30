import matplotlib
import pandas as pd
import streamlit as st
import seaborn as sns
from markdown_it.rules_core import inline

import matplotlib.pyplot as plt

import eda

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
