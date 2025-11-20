import pandas as pd
from pathlib import Path


def build_features(df, data_cfg: dict):
    """
    Aplica a engenharia de features no dataframe usando o bloco
    'data' do YAML (onde está o target_column).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe já pré-processado

    data_cfg : dict
        Subbloco do YAML: config["data"]
        Deve conter:
            - target_column : str
    """

    # -------------------------------------------------------------------
    # 1. Descobrindo a coluna target via dict do YAML
    # -------------------------------------------------------------------
    target_column = data_cfg.get("target_column")
    if target_column is None:
        raise ValueError("O YAML não possui 'data.target_column' definido.")

    # Copia e renomeia temporariamente
    df_temp = df.copy().rename(columns={target_column: "No_show"})

    # -------------------------------------------------------------------
    # 2. Limpeza e conversões
    # -------------------------------------------------------------------
    df_temp = (
        df_temp
        .assign(
            No_show=lambda x: x["No_show"].map({"No": 0, "Yes": 1}).astype(int),
            ScheduledDay=lambda x: pd.to_datetime(x["ScheduledDay"], errors="coerce", utc=True),
            AppointmentDay=lambda x: pd.to_datetime(x["AppointmentDay"], errors="coerce", utc=True),
        )
        .query("Age >= 0")
        .dropna(subset=["ScheduledDay", "AppointmentDay"])
    )

    # -------------------------------------------------------------------
    # 3. Feature engineering
    # -------------------------------------------------------------------
    df_temp = df_temp.assign(
        waiting_days=lambda x: (x["AppointmentDay"] - x["ScheduledDay"]).dt.days,
        appointment_weekday=lambda x: x["AppointmentDay"].dt.weekday,
        appointment_month=lambda x: x["AppointmentDay"].dt.month,
        hour_scheduled=lambda x: x["ScheduledDay"].dt.hour,
        hour_appointment=lambda x: x["AppointmentDay"].dt.hour,
        is_weekend=lambda x: x["AppointmentDay"].dt.weekday.isin([5, 6]).astype(int),
    )

    # -------------------------------------------------------------------
    # 4. Número de consultas anteriores do paciente
    # -------------------------------------------------------------------
    df_temp["previous_appointments"] = (
        df_temp.groupby("PatientId")["AppointmentID"].transform("count") - 1
    )

    # -------------------------------------------------------------------
    # 5. Taxa de no-show por paciente
    # -------------------------------------------------------------------
    df_temp["no_show_rate_patient"] = (
        df_temp.groupby("PatientId")["No_show"].transform("mean")
    )

    # -------------------------------------------------------------------
    # 6. Grupo de idade
    # -------------------------------------------------------------------
    age_bins = [0, 1, 13, 18, 60, df_temp["Age"].max() + 1]
    age_labels = ["baby", "child", "teen", "adult", "senior"]

    df_temp["age_group"] = pd.cut(
        df_temp["Age"],
        bins=age_bins,
        labels=age_labels,
        right=False
    ).astype(str)

    # -------------------------------------------------------------------
    # 7. Renomeia o target de volta ao nome original
    # -------------------------------------------------------------------
    df_temp = df_temp.rename(columns={"No_show": target_column})

    return df_temp
