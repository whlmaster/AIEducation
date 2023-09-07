import pandas as pd

# Список названий таблиц
tabs_names = ['D_clients', 'D_close_loan', 'D_job',
              'D_last_credit', 'D_loan', 'D_pens',
              'D_salary', 'D_target', 'D_work']

# импорт данных
[D_clients, D_close_loan, D_job,
 D_last_credit, D_loan, D_pens,
 D_salary, D_target, D_work] = [
    pd.read_csv(f'https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/{i}.csv') for i in
    tabs_names]


# перевести названия столбцов в нижний регистр
def lower_case(list):
    for i in list:
        i.columns = i.columns.str.lower().str.replace(' ', '_')


lower_case([D_clients, D_close_loan, D_job,
            D_last_credit, D_loan, D_pens,
            D_salary, D_target, D_work])

#предобработка
D_salary = D_salary.drop_duplicates()
for i in ['gen_industry', 'gen_title', 'job_dir']:
  D_job[i] = D_job[i].fillna("Нет инфо")
D_job['work_time'] = D_job['work_time'].fillna(0)
D_job = D_job[D_job['work_time'] < D_job['work_time'].quantile(0.99)]

D_clients = D_clients[
    ~(
        (D_clients['age']<D_clients.query('socstatus_pens_fl == 1')['age'].quantile(0.05)) & (D_clients['socstatus_pens_fl']==1)
        )
    ]

D_salary = D_salary[D_salary['personal_income']<D_salary['personal_income'].quantile(0.98)]

#объединение данных по кредитной истории
D_loan_hist = D_loan.merge(D_close_loan, how='inner', on='id_loan')
D_loan_sum = D_loan_hist.groupby('id_client').agg(
    {'id_loan': 'count', 'closed_fl': 'sum'})  # количество ссуд на одного клиента
D_loan_sum = D_loan_sum.rename(columns={"id_loan": "loans_count"})
D_clients = D_clients.rename(columns={"id": "id_client"})

# общая табилца на основании id клиента
tab_values = D_target.merge(D_loan_sum, how='inner', on='id_client')
tab_values = tab_values.merge(D_salary[['id_client', 'personal_income']], how='inner', on='id_client')
tab_values = tab_values.merge(
    D_clients[['id_client', 'age', 'socstatus_work_fl', 'socstatus_pens_fl', 'gender', 'child_total', 'dependants']],
    how='inner', on='id_client')
tab_values = tab_values.drop(labels=['id_client', 'agreement_rk'], axis=True)

# таблица добавлением групп для количественных переменных
tab_analize = tab_values.copy()
tab_analize['age_gr'] = pd.cut(tab_analize['age'], 10)
tab_analize['personal_income_gr'] = pd.cut(tab_analize['personal_income'], 8)
tab_analize['dependants_sum'] = tab_analize['dependants'] + tab_analize['child_total']
tab_analize['dependants_gr'] = pd.cut(tab_analize['dependants_sum'],
                                      [0, 2, 4, 6, 8, tab_analize['dependants_sum'].max()],
                                      right=False)

# добавить функции и графики, которые будут перенесены в app.py
