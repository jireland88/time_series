import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from statsmodels.regression.linear_model import WLS
from statsmodels.tsa.arima.model import ARIMA
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True


class Jodi:
    def __init__(self):
        self.frame = pd.read_csv("../datasets/jodi.csv")
        self.frame["TIME_PERIOD"] = pd.to_datetime(
            self.frame["TIME_PERIOD"], format="%Y-%m"
        )

    def get_series(
        self, ref_area: str, flow_breakdown: str, unit_measure: str = "M3"
    ) -> pd.Series:
        s = self.frame[
            (self.frame["REF_AREA"] == ref_area)
            & (self.frame["FLOW_BREAKDOWN"] == flow_breakdown)
            & (self.frame["UNIT_MEASURE"] == unit_measure)
        ].set_index("TIME_PERIOD")["OBS_VALUE"]

        s = s.divide(s.index.days_in_month).dropna()

        return s


def test_model(
    s: pd.Series, model: callable, window_size: int = 5 * 12, forward_steps: int = 12
) -> pd.DataFrame:
    first_start_train_date = s.index.min()
    last_start_train_date = s.index.max() - relativedelta(
        months=forward_steps + window_size + 1
    )

    assert (
        first_start_train_date + relativedelta(months=window_size + forward_steps)
        < s.index.max()
    )

    start_train_dates = pd.date_range(
        first_start_train_date, last_start_train_date, freq="MS"
    )

    errors = []

    for start_train_date in start_train_dates:
        end_train_date = start_train_date + relativedelta(months=window_size)

        start_test_date = end_train_date + relativedelta(months=1)
        end_test_date = end_train_date + relativedelta(months=forward_steps)

        train = s[start_train_date:end_train_date]
        test = s[start_test_date:end_test_date]

        pred = model(train, forward_steps)

        error = pd.Series(test - pred, name=start_test_date)
        error.index = list(range(1, forward_steps + 1))

        errors.append(error)

    errors = pd.concat(errors, axis=1).T

    return errors


#def bootstrap_pi(s: pd.Series, model: callable, alpha: float = 0.95):




def prophet_forecast(s: pd.Series, steps: int) -> pd.Series:
    past = s.to_frame("y").reset_index().rename(columns={s.index.name: "ds"})
    m = Prophet().fit(past)
    future = m.make_future_dataframe(periods=steps, freq="MS")
    forecast = m.predict(future)

    return forecast.set_index("ds")["yhat"].tail(12)


def wls(s: pd.Series, steps: int, decay: float = 0.8, return_residuals: bool = False) -> pd.Series:
    frame = s.to_frame("y")

    start_date = frame.index.min()
    frame["t"] = (frame.index - start_date).days

    for mth in range(1, 13):
        frame[f"month_{mth}"] = (frame.index.month == mth).astype(int)

    frame["weights"] = [1 * (decay) ** i for i in range(len(frame.index))][::-1]
    frame["1"] = 1

    model = WLS(
        frame["y"], frame.drop(["weights", "y"], axis=1), frame["weights"]
    ).fit()

    pred_index = pd.date_range(
        frame.index[-1] + relativedelta(months=1),
        frame.index[-1] + relativedelta(months=steps),
        freq="MS",
    )
    pred_frame = pd.DataFrame(index=pred_index, columns=["t"])
    pred_frame["t"] = (pred_frame.index - start_date).days

    for mth in range(1, 13):
        pred_frame[f"month_{mth}"] = (pred_frame.index.month == mth).astype(int)

    pred_frame["1"] = 1

    if return_residuals:
        return model.predict(pred_frame), model.resid

    return model.predict(pred_frame)


def wls_ar(s: pd.Series, steps: int, decay: float = 0.8) -> pd.Series:
    initial_forecast, resid = wls(s=s, steps=steps, decay=decay, return_residuals=True)

    error_model = ARIMA(
            endog=resid,
            order=(1, 0, 0)
    ).fit()
    error_pred = error_model.forecast(steps)

    forecast = initial_forecast + error_pred

    return forecast


if __name__ == "__main__":
    """
    Goal is to find the best time series library for forecasting series found in the JOJI Gas database
    """

    jodi = Jodi()

    time_series = {
        "Egypt Production": jodi.get_series("EG", "INDPROD"),
        "Egypt Demand": jodi.get_series("EG", "TOTDEMO"),
        "Algerian Production": jodi.get_series("DZ", "INDPROD"),
        "Algerian Demand": jodi.get_series("DZ", "TOTDEMO"),
        "Trinidad Production": jodi.get_series("TT", "INDPROD"),
        "GB Production": jodi.get_series("GB", "INDPROD"),
        "Australia Production": jodi.get_series("AU", "INDPROD"),
        "Australia Demand": jodi.get_series("AU", "TOTDEMO"),
        "India Production": jodi.get_series("IN", "INDPROD"),
        "India Demand": jodi.get_series("IN", "TOTDEMO"),
        "Japan Production": jodi.get_series("JP", "INDPROD"),
        "Japan Demand": jodi.get_series("JP", "TOTDEMO"),
        "Korea Production": jodi.get_series("KR", "INDPROD"),
        "Korea Demand": jodi.get_series("KR", "TOTDEMO"),
        "Thailand Production": jodi.get_series("TH", "INDPROD"),
    }

    models = {
#        "ARIMA": lambda s, steps: auto_arima(s).predict(steps),
        "ETS": lambda s, steps: ExponentialSmoothing(
            s, seasonal="add", trend="add", seasonal_periods=12, freq="MS"
        )
        .fit()
        .forecast(steps),
#        "SARIMA": lambda s, steps: auto_arima(s, m=12).predict(steps),
#        "Prophet": prophet_forecast,
        "WLS": wls,
        "WLS AR": wls_ar
    }

    forward_steps = 12
    results = []

    for ts_name, ts in time_series.items():
        for model_name, model in models.items():
            print(f"Testing model: {model_name} on time series: {ts_name}")
            errors = test_model(ts, model, forward_steps=forward_steps)

            # Compute some summary stats on the errors
            for step in range(1, forward_steps + 1):
                errors_step = errors[step]

                bias = errors_step.mean()
                rmse = (errors_step**2).mean() ** (1 / 2)

                # Normalise so can compare across different time series
                bias = (bias / ts.mean()) * 100
                rmse = (rmse / ts.mean()) * 100

                results.append(
                    {
                        "Time Series": ts_name,
                        "n": len(ts),
                        "Model": model_name,
                        "Forward Steps": step,
                        "Bias (%)": bias,
                        "RMSE (%)": rmse,
                    }
                )

    results = pd.DataFrame(results)

    for var in ["Bias (%)", "RMSE (%)"]:
        sns.catplot(
            data=results,
            x="Model",
            y=var,
            col="Forward Steps",
            kind="violin",
            col_wrap=2,
        ).fig.savefig(f"../plots/results_{var}.png")
