# Models 
oogway => neural network <br/> shifu => linear regression
# Example Request
  **Route:**
      http://services.amozeshgam.com:8000/ai/planning
  <br/>
  **Header Data:** 
      add 'x-api-key' and add 'model'
  <br/>
  **Json Data:** <br/>
  ```json
      {
        "age":"18",
        "educationalStatus":"0",//Studying = 0 graduate = 1
        "fieldOfStudy":"1",//computer = 1 more = 0
        "maritalStatus":"0",//married = 1 single = 0
        "gender":"1",//man = 1 woman = 0
        "militaryStatus":"1",//Educational exemption = 1  exemption = 2 runaway = 3 
        "freeTime":"2",//with hours
        "targetIncome":"14",//with milion
        "intentionToMigrate":"0",//true = 1 false = 0
        "interestInMathematics":"0",//true = 1 false = 0
        "computerExperience":"1",//with year
        "whichOneDoYouLikeMore":"0",//engineering = 1 artistic = 0
        "whichCaseIsmoreRelevant":"0",//engineering = 1 artistic = 0
        "doYouWorkOnHolidays":"1",//true = 1 false = 0
        "disability":"0",//true = 1 false = 0
        "addictionred":"0"//true = 1 false = 0
      }
  ```
# Errors
  - error12 => not found model name
  - errorz => not found api key
  - error0 => error in predict ai model
