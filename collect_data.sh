
# Created data folder
mkdir -p ./data

#### The following code is used to download the partially preprocessed datasets from https://github.com/steven7woo/fair_regression_reduction

## Download Adult dataset
wget -P ./data https://raw.githubusercontent.com/steven7woo/fair_regression_reduction/master/data/adult_full.csv

## Download Communities and Crime Data Set dataset
wget -P ./data https://raw.githubusercontent.com/steven7woo/fair_regression_reduction/master/data/communities.csv

## Download Law School dataset
wget -P ./data https://raw.githubusercontent.com/steven7woo/fair_regression_reduction/master/data/lawschool.csv


# #### Uncomment the following lines to download the datasets Adult and C&C used in the paper from raw data and converts it to csv format

# ## Collect Adult data from UCI ML repository (https://archive.ics.uci.edu/ml/datasets/adult)
# wget -P ./data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

# ### convert to csv and add header
# cp ./data/adult.data ./data/adult_full.csv
# header=
# sed -i '1s/^/"age","workclass","fnlwgt","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week","native.country","income"\n/' data/adult_full.csv 

# ## Collect Communities and Crime Data Set data from UCI ML repository (https://archive.ics.uci.edu/ml/datasets/communities+and+crime)
# wget -P ./data https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data

# ### convert to csv and add header
# cp ./data/communities.data ./data/communities_full.csv
# sed -i '1s/^/state,county,community,communityname,fold,population,householdsize,racepctblack,racePctWhite,racePctAsian,racePctHisp,agePct12t21,agePct12t29,agePct16t24,agePct65up,numbUrban,pctUrban,medIncome,pctWWage,pctWFarmSelf,pctWInvInc,pctWSocSec,pctWPubAsst,pctWRetire,medFamInc,perCapInc,whitePerCap,blackPerCap,indianPerCap,AsianPerCap,OtherPerCap,HispPerCap,NumUnderPov,PctPopUnderPov,PctLess9thGrade,PctNotHSGrad,PctBSorMore,PctUnemployed,PctEmploy,PctEmplManu,PctEmplProfServ,PctOccupManu,PctOccupMgmtProf,MalePctDivorce,MalePctNevMarr,FemalePctDiv,TotalPctDiv,PersPerFam,PctFam2Par,PctKids2Par,PctYoungKids2Par,PctTeen2Par,PctWorkMomYoungKids,PctWorkMom,NumIlleg,PctIlleg,NumImmig,PctImmigRecent,PctImmigRec5,PctImmigRec8,PctImmigRec10,PctRecentImmig,PctRecImmig5,PctRecImmig8,PctRecImmig10,PctSpeakEnglOnly,PctNotSpeakEnglWell,PctLargHouseFam,PctLargHouseOccup,PersPerOccupHous,PersPerOwnOccHous,PersPerRentOccHous,PctPersOwnOccup,PctPersDenseHous,PctHousLess3BR,MedNumBR,HousVacant,PctHousOccup,PctHousOwnOcc,PctVacantBoarded,PctVacMore6Mos,MedYrHousBuilt,PctHousNoPhone,PctWOFullPlumb,OwnOccLowQuart,OwnOccMedVal,OwnOccHiQuart,RentLowQ,RentMedian,RentHighQ,MedRent,MedRentPctHousInc,MedOwnCostPctInc,MedOwnCostPctIncNoMtg,NumInShelters,NumStreet,PctForeignBorn,PctBornSameState,PctSameHouse85,PctSameCity85,PctSameState85,LemasSwornFT,LemasSwFTPerPop,LemasSwFTFieldOps,LemasSwFTFieldPerPop,LemasTotalReq,LemasTotReqPerPop,PolicReqPerOffic,PolicPerPop,RacialMatchCommPol,PctPolicWhite,PctPolicBlack,PctPolicHisp,PctPolicAsian,PctPolicMinor,OfficAssgnDrugUnits,NumKindsDrugsSeiz,PolicAveOTWorked,LandArea,PopDens,PctUsePubTrans,PolicCars,PolicOperBudg,LemasPctPolicOnPatr,LemasGangUnitDeploy,LemasPctOfficDrugUn,PolicBudgPerPop,ViolentCrimesPerPop\n/' data/communities_full.csv
# ### remove first 5 columns
# cut -d, -f1-5 ./data/communities_full.csv --complement > ./data/communities.csv
