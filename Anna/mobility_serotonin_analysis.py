import pandas as pd
from scipy.stats import ttest_ind

def load_and_merge_data(control_mobility_paths, control_serotonin_paths, pcb_mobility_paths, pcb_serotonin_paths):
    def load_data(paths):
        return [pd.read_csv(path) for path in paths]

    control_mobility_dfs = load_data(control_mobility_paths)
    control_serotonin_dfs = load_data(control_serotonin_paths)
    pcb_mobility_dfs = load_data(pcb_mobility_paths)
    pcb_serotonin_dfs = load_data(pcb_serotonin_paths)

    for df in control_serotonin_dfs + pcb_serotonin_dfs:
        df['Time (s)'] = df['Time (s)'].round(2)

    merged_control_dfs = [pd.merge(mob_df, ser_df, on='Time (s)', how='inner')
                          for mob_df, ser_df in zip(control_mobility_dfs, control_serotonin_dfs)]
    merged_pcb_dfs = [pd.merge(mob_df, ser_df, on='Time (s)', how='inner')
                      for mob_df, ser_df in zip(pcb_mobility_dfs, pcb_serotonin_dfs)]

    merged_control_df = pd.concat(merged_control_dfs)
    merged_pcb_df = pd.concat(merged_pcb_dfs)

    return merged_control_df, merged_pcb_df, merged_control_dfs, merged_pcb_dfs

def extract_and_compare_serotonin(merged_control_df, merged_pcb_df):
    control_mobile_serotonin = merged_control_df[merged_control_df['mob'] == 0]['Z-score']
    control_immobile_serotonin = merged_control_df[merged_control_df['mob'] == 1]['Z-score']
    pcb_mobile_serotonin = merged_pcb_df[merged_pcb_df['mob'] == 0]['Z-score']
    pcb_immobile_serotonin = merged_pcb_df[merged_pcb_df['mob'] == 1]['Z-score']

    t_stat_mobile, p_value_mobile = ttest_ind(control_mobile_serotonin, pcb_mobile_serotonin)
    t_stat_immobile, p_value_immobile = ttest_ind(control_immobile_serotonin, pcb_immobile_serotonin)

    print(f"T-test for mobile state: t-statistic = {t_stat_mobile}, p-value = {p_value_mobile}")
    print(f"T-test for immobile state: t-statistic = {t_stat_immobile}, p-value = {p_value_immobile}")

    return (control_mobile_serotonin, control_immobile_serotonin, pcb_mobile_serotonin, pcb_immobile_serotonin)

def calculate_average_serotonin(merged_control_dfs, merged_pcb_dfs):
    def calculate_mean(df, state):
        return df[df['mob'] == state]['Z-score'].mean()

    averages = []
    for control_df, pcb_df in zip(merged_control_dfs, merged_pcb_dfs):
        averages.append({
            'control_mobile': calculate_mean(control_df, 0),
            'control_immobile': calculate_mean(control_df, 1),
            'pcb_mobile': calculate_mean(pcb_df, 0),
            'pcb_immobile': calculate_mean(pcb_df, 1)
        })

    overall_averages = {
        'control_mobile': sum(avg['control_mobile'] for avg in averages) / len(averages),
        'control_immobile': sum(avg['control_immobile'] for avg in averages) / len(averages),
        'pcb_mobile': sum(avg['pcb_mobile'] for avg in averages) / len(averages),
        'pcb_immobile': sum(avg['pcb_immobile'] for avg in averages) / len(averages)
    }

    return averages, overall_averages
