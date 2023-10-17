
import numpy as np
import pandas as pd

from heat_alerts.online_rl.datautils import load_rl_states_by_county

counties = [41067, 53015, 20161, 37085, 48157,
             28049, 19153, 17167, 31153, 6071, 4013,
             34021, 19155, 17115, 29021, 29019, 5045, 40017, 21059,
             47113, 42017, 22109, 45015, 13031, 48367, 22063, 41053,
             32003, 4015, 6025]

years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
n_days = 153

for county in counties:
    base_dict, effect_dict, extra_dict, other_dict = load_rl_states_by_county(
        county,
        "data/processed",
        years,
        match_similar=True,
        as_tensors=False,
        incorp_forecasts=True,
    )

    for q in ['q50', 'q60', 'q70', 'q80', 'q90', 'q100']:
        MAE = np.zeros((len(years), len(extra_dict[q].columns)))
        i = 0
        for y in years:
            COI = extra_dict[q].loc[(county,y)]
            others = extra_dict[q].drop((county,y), axis=0)
            diffs = np.absolute(np.subtract(others, COI))
            mae = diffs.mean(axis=0)# /COI
            MAE[i,:] = mae
            i+= 1
        avg_MAE = MAE.mean(axis=0)
        pd.DataFrame(avg_MAE).to_csv("F_accuracy/" + q + "_fips-" + str(county) + ".csv")
        print(q)
    
    print(county)


for q in ['q50', 'q60', 'q70', 'q80', 'q90', 'q100']:
    MAE = np.zeros((len(counties), n_days))
    i = 0
    for county in counties:
        mae = pd.read_csv("F_accuracy/" + q + "_fips-" + str(county) + ".csv")
        mae.columns = ["index", "mae"]
        MAE[i,:] = mae["mae"]
        i+=1
    avg_MAE = MAE.mean(axis=0)
    pd.DataFrame(avg_MAE).to_csv("F_accuracy/" + q + ".csv")
    print(q)

