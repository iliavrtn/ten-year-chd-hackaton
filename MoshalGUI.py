import PySimpleGUI as sg
from collections import namedtuple
import os, sys, ast
import threading
import predict_sample as AI_Model

import run_ce_optimization as optimizer


# Disable
def blockPrint():
    orgStdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return orgStdout


# Restore
def enablePrint(orgStd):
    sys.stdout = orgStd  # sys.__stdout__


INI_FILE_NAME = 'Default_Ini.ini'
GUI_ICON_PATH = './Images/MoshalLogo.ico'
GUI_HACKATON_BAR_PATH = './Images/MoshalHackatonBar.png'

EducationList = ['No Education', 'High School', 'Student', 'University/College']
SexList = ['Male', 'Female']
YesNoList = ['Yes', 'No']

ParamData = namedtuple('ParamData', ['paramName', 'paramInputKey', 'paramValidityKey'])
Params_Map = {
    'id': ParamData(f'ID', f'-ID_Input-', f'-|Check|ID_Validity-'),
    'age': ParamData(f'Age', f'-Age_Input-', f'-|Check|Age_Validity-'),
    'Education': ParamData(f'Education', f'|Education|', f'-|Check|Education_Validity-'),
    'Sex': ParamData(f'Sex', f'|Sex|', f'-|Check|Sex_Validity-'),
    'isSmoking': ParamData(f'Smoking?', f'|isSmoking|', f'-|Check|isSmoking_Validity-'),
    'cigsPerDay': ParamData(f'How Much Cigarets Per Day?', f'-cigsPerDay_Input-', f'-|Check|cigsPerDay_Validity-'),
    'BP_Meds': ParamData(f'Are you using Blood Pressure Medications?', f'|BP_Meds|', f'-|Check|BP_Meds_Validity-'),
    'prevalentStroke': ParamData(f'Did you had a Heart Stroke in the past?', f'|prevalentStroke|',
                                 f'-|Check|prevalentStroke_Validity-'),
    'prevalentHyp': ParamData(f'Did you have Hypertensive', f'|prevalentHyp|', f'-|Check|prevalentHyp_Validity-'),
    'hasDiabetes': ParamData(f'Do you have Diabetes?', f'|hasDiabetes|', f'-|Check|hasDiabetes_Validity-'),
    'totalCholesterol': ParamData(f'Your Cholesterol Level', f'-totalCholesterol_Input-',
                                  f'-|Check|totalCholesterol_Validity-'),
    'sysBP': ParamData(f'Systolic Blood Pressure', f'-Param_Input-', f'-|Check|sysBP_Validity-'),
    'diaBP': ParamData(f'Diastolic Blood Pressure', f'-diaBP_Input-', f'-|Check|diaBP_Validity-'),
    'BMI': ParamData(f'Your BMI', f'-BMI_Input-', f'-|Check|BMI_Validity-'),
    'heartrate': ParamData(f'Your Heartrate', f'-heartrate_Input-', f'-|Check|heartrate_Validity-'),
    'glucose': ParamData(f'Your Glucose', f'-glucose_Input-', f'-|Check|glucose_Validity-')
}


# global optResult = ''

def optThread(sample):
    result = optimizer.run_ce_optimization(sample)
    print(result)


def convertSex(sexStr):
    if 'Male' == sexStr:
        return 'M'
    elif 'Female' == sexStr:
        return 'F'


def convertEducation(eduStr):
    if 'No Education' == eduStr:
        return 1
    elif 'High School' == eduStr:
        return 2
    elif 'Student' == eduStr:
        return 3
    elif 'University/College' == eduStr:
        return 4


def converYesNo(answerStr):
    if 'Yes' == answerStr:
        return 1
    elif 'No' == answerStr:
        return 0


def load_ini(window):
    if os.path.exists(INI_FILE_NAME):
        with open(INI_FILE_NAME, 'r') as text_file:
            out = text_file.read()
            new_values = ast.literal_eval(out)
            for new_value in new_values:
                update_window(window, new_value, new_values[new_value])


def save_ini(values):
    with open(INI_FILE_NAME, 'w+') as text_file:
        text_file.write(str(values))


def update_window(window, value_key, new_value):
    if 'Browse' not in value_key:
        window.Element(value_key).Update(new_value)
        window.Refresh()


def checkValidityByParam(values):
    errorMsg = ''

    if int(values[Params_Map['id'].paramInputKey]) <= 0:
        errorMsg += 'Given ID is invalid, should be a possitive number\n'
    if not (0 < int(values[Params_Map['age'].paramInputKey]) < 120):
        errorMsg += 'Given Age is invalid, should be a possitive number\n'
    if values[Params_Map['Education'].paramInputKey] == '':
        errorMsg += 'You should choose your education level\n'
    if values[Params_Map['Sex'].paramInputKey] == '':
        errorMsg += 'You should declare your sex\n'
    if values[Params_Map['isSmoking'].paramInputKey] == '':
        errorMsg += 'You should declare if you are smoking or not\n'
    if values[Params_Map['isSmoking'].paramInputKey] == 'Yes' and int(
            values[Params_Map['cigsPerDay'].paramInputKey]) <= 0:
        errorMsg += 'You should say how much Cigaret you are smoking per day\n'
    if values[Params_Map['isSmoking'].paramInputKey] == 'No' and int(
            values[Params_Map['cigsPerDay'].paramInputKey]) > 0:
        errorMsg += 'You said that you are not smoking, sure about that?\n'
    if int(values[Params_Map['cigsPerDay'].paramInputKey]) < 0:
        errorMsg += 'Cigarets amount can\'t be negative\n'
    if values[Params_Map['BP_Meds'].paramInputKey] == '':
        errorMsg += 'You should answer if you\'re taking Blood Pressure Medication or not\n'
    if values[Params_Map['prevalentHyp'].paramInputKey] == '':
        errorMsg += 'You Should answer if you had a Hypertensive\n'
    if values[Params_Map['hasDiabetes'].paramInputKey] == '':
        errorMsg += 'You should say if you are diagnosed with Diabetes\n'
    if int(values[Params_Map['totalCholesterol'].paramInputKey]) <= 0:
        errorMsg += 'You should say what is you cholesterol level\n'
    if float(values[Params_Map['diaBP'].paramInputKey]) <= 40:
        errorMsg += 'Your Systolic Blood Pressure is invalid\n'
    if float(values[Params_Map['sysBP'].paramInputKey]) <= float(values[Params_Map['diaBP'].paramInputKey]):
        errorMsg += 'Your Distolic Blood Pressure is invalid\n'
    if float(values[Params_Map['BMI'].paramInputKey]) <= 0:
        errorMsg += 'Given BMI is invalid, should be the result of (weight)/(heigth^2)\n'
    if int(values[Params_Map['heartrate'].paramInputKey]) <= 0:
        errorMsg += 'Given Heartrate is invalid\n'
    if int(values[Params_Map['glucose'].paramInputKey]) <= 0:
        errorMsg += 'GIven Glucose level is invalid\n'
    if values[Params_Map['prevalentStroke'].paramInputKey] == '':
        errorMsg += 'We nee your info about if you had a stroke in the past\n'

    return errorMsg


class MoshalGUI(sg.Window):
    def __init__(self):
        self.initUI()
        sg.theme('DarkTeal12')
        super().__init__('Diabetes Possible Damage Prediction', self.mainLayout, resizable=True, finalize=True,
                         icon=GUI_ICON_PATH)

        load_ini(self)
        self.Finalize()

    def initUI(self):
        self.outputColumn = [
            [sg.Output(size=(50, 20), font='Courier 12')]
        ]

        self.insertDataColumn = [
            [sg.Text(Params_Map['id'].paramName, size=(33, 1)),
             sg.Input(key=Params_Map['id'].paramInputKey, size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['age'].paramName, size=(33, 1)),
             sg.Input(key=Params_Map['age'].paramInputKey, size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['Education'].paramName, size=(33, 1)),
             sg.Combo(EducationList, key='|Education|', size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['Sex'].paramName, size=(33, 1)), sg.Combo(SexList, key='|Sex|', size=(20, 1)),
             sg.Push()],
            [sg.Text(Params_Map['isSmoking'].paramName, size=(33, 1)),
             sg.Combo(YesNoList, key='|isSmoking|', size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['cigsPerDay'].paramName, size=(33, 1)),
             sg.Input(key=Params_Map['cigsPerDay'].paramInputKey, size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['BP_Meds'].paramName, size=(33, 1)), sg.Combo(YesNoList, key='|BP_Meds|', size=(20, 1)),
             sg.Push()],
            [sg.Text(Params_Map['prevalentStroke'].paramName, size=(33, 1)),
             sg.Combo(YesNoList, key='|prevalentStroke|', size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['prevalentHyp'].paramName, size=(33, 1)),
             sg.Combo(YesNoList, key='|prevalentHyp|', size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['hasDiabetes'].paramName, size=(33, 1)),
             sg.Combo(YesNoList, key='|hasDiabetes|', size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['totalCholesterol'].paramName, size=(33, 1)),
             sg.Input(key=Params_Map['totalCholesterol'].paramInputKey, size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['sysBP'].paramName, size=(33, 1)),
             sg.Input(key=Params_Map['sysBP'].paramInputKey, size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['diaBP'].paramName, size=(33, 1)),
             sg.Input(key=Params_Map['diaBP'].paramInputKey, size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['BMI'].paramName, size=(33, 1)),
             sg.Input(key=Params_Map['BMI'].paramInputKey, size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['heartrate'].paramName, size=(33, 1)),
             sg.Input(key=Params_Map['heartrate'].paramInputKey, size=(20, 1)), sg.Push()],
            [sg.Text(Params_Map['glucose'].paramName, size=(33, 1)),
             sg.Input(key=Params_Map['glucose'].paramInputKey, size=(20, 1)), sg.Push()],
            [sg.Push(), sg.Button('Check Input Validate', key='-Check_Validity-'),
             sg.Button('Run Check', key='-Run_Optimizer-'), sg.Push()],
            []
        ]

        self.explanationColumn = [
            [sg.Push(), sg.Text('Title'), sg.Push()],
            [sg.Text('BasicInput', size=(15, 1)), sg.Input(key='-BasicInputField-'), sg.FileBrowse()],
            []
        ]

        self.mainLayout = [
            [sg.Push(), sg.Image(GUI_HACKATON_BAR_PATH, size=(1000, 202)), sg.Push()],
            [sg.Column(self.insertDataColumn), sg.VSeperator(), sg.Column(self.outputColumn)]
        ]

    def callOptimizer(self, values):
        sample = [
            int(values[Params_Map['id'].paramInputKey]),
            int(values[Params_Map['age'].paramInputKey]),
            convertEducation(values[Params_Map['Education'].paramInputKey]),
            convertSex(values[Params_Map['Sex'].paramInputKey]),
            converYesNo(values[Params_Map['isSmoking'].paramInputKey]),
            int(values[Params_Map['cigsPerDay'].paramInputKey]),
            converYesNo(values[Params_Map['BP_Meds'].paramInputKey]),
            converYesNo(values[Params_Map['prevalentStroke'].paramInputKey]),
            converYesNo(values[Params_Map['prevalentHyp'].paramInputKey]),
            converYesNo(values[Params_Map['hasDiabetes'].paramInputKey]),
            int(values[Params_Map['totalCholesterol'].paramInputKey]),
            float(values[Params_Map['sysBP'].paramInputKey]),
            float(values[Params_Map['diaBP'].paramInputKey]),
            float(values[Params_Map['BMI'].paramInputKey]),
            int(values[Params_Map['heartrate'].paramInputKey]),
            int(values[Params_Map['glucose'].paramInputKey])
        ]
        (riskPrediction, ProbList) = AI_Model.predict_sample(sample)
        predResultAnswer = 'High' if riskPrediction else 'Low'
        print(f'{predResultAnswer} risk for CHD complication.')
        print(f'Probabilities are:\nFor low risk: {ProbList[0]:.2f}\nFor high risk: {ProbList[1]:.2f}')

        if riskPrediction:
            print("\nCalculating...\n")
            threading.Thread(target=optThread, args=(sample,), daemon=True).start()

    def executeButtonClick(self, event, values):
        if event == '-Check_Validity-':
            errorMsg = checkValidityByParam(values)
            if errorMsg != '':
                print(errorMsg)

        elif event == '-Run_Optimizer-':
            isInputValid = checkValidityByParam(values)
            if '' == isInputValid:
                self.callOptimizer(values)
            else:
                print(isInputValid)
                print('Check Input! Process Terminated!')

    def mainloop(self):
        try:
            while True:
                event, values = self.read()

                if values:
                    save_ini(values)

                if event == sg.WIN_CLOSED or event == 'Exit':
                    break

                self.executeButtonClick(event, values)

                self.Refresh()

        except Exception as e:
            sg.Print('Exception in my event loop for the program:', sg.__file__, e, keep_on_top=True, wait=True)

        self.close()


if '__main__' == __name__:
    myWindow = MoshalGUI()
    myWindow.mainloop()
