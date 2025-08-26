import pandas as pd

first_file_path = 'ctr_sample_submission.csv'
df_first = pd.read_csv(first_file_path)

second_file_path = 'sample_submission.csv'  
df_second = pd.read_csv(second_file_path)

if len(df_first) != len(df_second):
    raise ValueError("Количество строк в файлах не совпадает!")

df_first.iloc[:, 1] = df_second.iloc[:, 1]
output_file_path = 'updated_ctr_sample_submission.csv'
df_first.to_csv(output_file_path, index=False)

print(f"Данные успешно обновлены и сохранены в {output_file_path}")
