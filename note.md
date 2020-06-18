### Expectations
- Development, implement, application of real world data
- Take one of the listed code and continue it's development **OR** Start from scratch

### Targets
- Understand the modeling
- Add new features and funtionality to the code or theory
- Increase score, detail, improve results and insights 
- Results should be deliverable( meaning sufficient **Documentations** and **Commenting**)
- Use integrated develpoment environment and version control
- Use your code and new models to test out your hypothesis and scenarios. Validate using existing data.
- **Plan and document all tests and modeling experiements** using redmine
- Progress checkup every friday. 1 page pdf. i inch margin, 11pt, Times New Roman, 1.5 spacing. Regarding progress and next week's anticipation
- 

## Sources 
- [Epidemic Modeling](https://medium.com/data-for-science/epidemic-modeling-101-or-why-your-covid19-exponential-fits-are-wrong-97aa50c55f8)
<br>
1. we can divide the population into different compartments representing the different stages of the disease and use the relative size of each compartment to model how the numbers evolve in time.
2. **Susceptible-Infected model (the healthy vs the infected)** : The dynamics is also simple, when a healthy person comes in contact with an infectious person s/he becomes infected with a given probability. Looks like a logistic regression model given that in an certain amount of time everyone is infected
3. **Susceptible-Infectious-Recovered** model
4. **Flattening the curve** : flattening the infected curve (to perhaps under the healthcare capacity curve)
5. Herd immunity
6. Incubation period(latent period)
7. COVID19 -> the case of **asymptomatic**(no symptons) patients is believed to be around 40%(or higher)
8. Delay in detection due to patients believing that they're not sick enough, this causes a delay in detection in a given country(US) which also causes an overestimation of the severity
9. published numbers are also cumulative. Making the number look alot bigger
10. It takes time to develop and distribute accurate tests. -> Real cases vs testing/tested cases
11. **Dynamic Lags** : This implies that there is a natural lag between the peak of new infections and the peak in the total number of infectious individuals that is proportional to the duration of the infectious period.
12. **Lockdown Procedures** : As we can see, a prematurely broken lockdown quickly results in a second wave of the epidemic leading to almost as many total cases as if there had been no intervention whatsoever. However, it does still have the benefit of keeping the peak number of sick individuals below what would normally be and a “spreading out” of the epidemic curve: In other words, the flattening of the curve that will help prevent the overwhelming of the healthcare system.
13. **Structured Populations** 

- [Epidemic Modeling 2 - some models are useful?](https://medium.com/data-for-science/epidemic-modeling-102-all-covid-19-models-are-wrong-but-some-are-useful-c81202cc6ee9)

1. 


- [Epidemic Modeling 3 - Confindence Intervals and Stochastic Effects](https://medium.com/data-for-science/epidemic-modeling-103-adding-confidence-intervals-and-stochastic-effects-to-your-covid-19-models-be618b995d6b)

- [Epidemic Modeling 4 - Seasonal Effects](https://medium.com/data-for-science/epidemic-modeling-104-impact-of-seasonal-effects-on-covid-19-16a1b14056f)


- [COVID-19 Data Source](https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model)

- [Building your own COVID-19 Model](https://towardsdatascience.com/building-your-own-covid-19-epidemic-simple-model-using-python-e39788fbda55)

- [COVID-19 Visualization](https://towardsdatascience.com/visualise-covid-19-case-data-using-python-dash-and-plotly-e58feb34f70f)

[Use this as reference](#https://github.com/a2975667/QV-app)

