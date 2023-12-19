def calculate_framingham_score(age, sex, is_smoking, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,
                               totChol, sysBP, diaBP, BMI, heartRate, glucose, education):
    # Define coefficients for the Framingham Risk Score
    coefficients = {
        "age": 0.0799,
        "sex": 0.6089,
        "is_smoking": 0.6536,
        "cigsPerDay": 0.0202,
        "BPMeds": 0.6948,
        "prevalentStroke": 0.9396,
        "prevalentHyp": 0.2402,
        "diabetes": 0.8507,
        "totChol": 0.0026,
        "sysBP": 0.0186,
        "diaBP": 0.0082,
        "BMI": 0.0622,
        "heartRate": 0.0144,
        "glucose": 0.0072,
        "education": 0.0015
    }

    # Calculate the Framingham Risk Score
    log_age = 0.0799 * age
    log_sex = 0.6089 * sex
    log_smoking = 0.6536 * is_smoking
    log_cigsPerDay = 0.0202 * cigsPerDay
    log_BPMeds = 0.6948 * BPMeds
    log_prevalentStroke = 0.9396 * prevalentStroke
    log_prevalentHyp = 0.2402 * prevalentHyp
    log_diabetes = 0.8507 * diabetes
    log_totChol = 0.0026 * totChol
    log_sysBP = 0.0186 * sysBP
    log_diaBP = 0.0082 * diaBP
    log_BMI = 0.0622 * BMI
    log_heartRate = 0.0144 * heartRate
    log_glucose = 0.0072 * glucose
    log_education = 0.0015 * education

    log_odds = (
            log_age +
            log_sex +
            log_smoking +
            log_cigsPerDay +
            log_BPMeds +
            log_prevalentStroke +
            log_prevalentHyp +
            log_diabetes +
            log_totChol +
            log_sysBP +
            log_diaBP +
            log_BMI +
            log_heartRate +
            log_glucose +
            log_education
    )

    # Calculate the 10-year risk of CHD using the Framingham Risk Score formula
    risk = 1 - (0.9402 ** (age - 45)) ** (2.7641 * (sex - 0.4176) + log_odds)

    return risk


