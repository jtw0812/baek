import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb


columns_탈락 = ['스트레스_총점','불안장애_총점','우울증_총점','알콜중독_총점','아동학대경험_총점']
columns_model = ["상담횟수",'나이','진단평가실시과목수',
                 "대학예배출석횟수"]

#범주형 변수
one_hot_column = ['성별']

##각 검사별 컬럼
UACI_column = ['Economicfactors-factors_Subjective', 'Interactionfactors-factors_Social', 'Psychologicalfactors-factors_self-regulation',
       'Psychologicalfactors-factors_Self-confidence', 'Psychologicalfactors-factors_relative', 'Economicfactors-factors_Physical',
       'Socialfactors-factors_level', 'Interactionfactors-factors_interpersonal', 'Psychologicalfactors-factors_career',
       'Psychologicalfactors-factors_Achievement', 'Interactionfactor-factor_Intimate', 'Educationalenvironmentfactors-factor__environment']

REST_4_column = ['대표 부적응 \n위험요인 수', '종합 적응도', '종합 위험도', '자살', '자해', '학교중단',
        '이타성', '친밀감', '대인관계기술', '유쾌함',
       '리더십', '적극성', '낙관성', '진솔함', '용기', '감수성', '대처유연성', '호기심', '성취욕구',
       '지적 탐구', '창의력', '끈기', '자기조절', '신중함', '계획성', '책임감', '감사', '포용', '지혜',
       '충동성', '수줍음', '스트레스 취약성', '자신감', '자아존중감', '자기비난',
       '진로효능감', '학업효능감','중독',
       '우울', '불안', '대학만족도', '전공만족도', '경제적 안정성', '교수와의 관계', '사회적 지지']
교수학습유형_column = ['반항', '완벽', '고군', '잡념', '만족', '외곬', '행동', '규범', '탐구', '이상']


def get_student_num(df):
    stu_num = df['학번']
    return stu_num
def get_num_no_exam(df):
    df['검사비수행갯수'] = 0

    for column_list in [columns_탈락, UACI_column, 교수학습유형_column, REST_4_column,['진단평가실시과목수']]:
        # 해당 검사의 모든 열 중 하나라도 NaN인 경우에 합산
        df['검사비수행갯수'] += df[column_list].isnull().any(axis=1).astype(int)

    df["상담횟수"].fillna(0, inplace= True)
    df['대학예배출석횟수'].fillna(0,inplace = True)

    return df
    

def one_hot(df):
    one_hot_encoded = pd.get_dummies(df[one_hot_column], prefix='one_hot').astype(float)
    df.drop(one_hot_column, axis = 1, inplace=True)
    df = pd.concat([df, one_hot_encoded], axis=1)
    return df

def drop_col(df):
    df = df[REST_4_column + UACI_column + columns_model + columns_탈락 + ['one_hot_남','one_hot_여','검사비수행갯수']]
    return df

def knn_impute_scale(df):
    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    imputer.fit(df)
    Xtrans = imputer.transform(df)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(Xtrans), columns=df.columns)
    return df_scaled

def predict(df,df_scaled, stu_num):
    filename = './정원외_포함.model'
    new_xgb_model = xgb.XGBClassifier() # 모델 초기화
    new_xgb_model.load_model(filename)
    x_prob = new_xgb_model.predict_proba(df_scaled)
    df['예측'] = x_prob[:,1]
    df['학번'] = stu_num
    break_points = [df['예측'].quantile(i) for i in [0.1, 0.2, 0.3, 0.4]]
    labels = ['위험', '경계', '주의', '관심', '안전']  # 5개 그룹 레이블
    def assign_group(value):
        if value <= break_points[0]:
            return labels[0]
        elif value <= break_points[1]:
            return labels[1]
        elif value <= break_points[2]:
            return labels[2]
        elif value <= break_points[3]:
            return labels[3]
        else:
            return labels[4]
    # 각 값에 대한 그룹 결정

    df['그룹'] = df['예측'].apply(assign_group)
    result = df[['학번','그룹','예측']]
    return result, df

df = pd.read_excel("./test_data.xlsx")

stu_num = get_student_num(df)
df = df[교수학습유형_column + REST_4_column +UACI_column + one_hot_column + columns_model + columns_탈락]
df = get_num_no_exam(df)
df = one_hot(df)
df = drop_col(df)
df_scaled = knn_impute_scale(df)
result, df = predict(df,df_scaled,stu_num)
result.to_excel('./정원외_포함_결과1.xlsx')
df.to_excel('./정원외_포함_결과2.xlsx')