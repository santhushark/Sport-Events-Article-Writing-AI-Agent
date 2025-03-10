# Sport-Events-Article-Writing-AI-Agent
Leveraged Agentic RAG workflow and FastAPI to build a backend Web Application that can write article on Sport events. 

This is a multi-agentic RAG application following Supervior Design Pattern. Implemented Web search feature using Tavily to gather relevant information and Human in the loop to evaluate generated articles and confirm using checkpointers. And finally the application is conatinerised.
+ The workflow starts from the "HumanWorkflow" agent which call the "ArticleChef" agent to carry of the article creation task
+ The "ArticleChef" agent which is basically the Supervisor agent checks if all the required information if present in the query to generate the article Eg: Tournament, Sport and the teams involved
+ Then in call for the "WebSearchQueryGeneratorAgent" to generate a web search query. The generated query is returned to the "ArticleChef"
+ The "ArticleChef" then calls the "WebSearch" agent which will perform web search to gather all the required information required to write the article. The information is then sent back to "ArticleChef"
+ The "ArticleChef" then calls the "ArticleWriter" agent which then analyses the information gathered from web and writes a beautiful sports article. The article is then retured to "ArticleChef"
+ The article is now sent back to "HumanWorkflow" where the article is evaluated by human and finally confirmed for publication.

## Steps to run the application in your local computer
+ Clone the repository and open the project in the IDE of your choice.
+ Create a virtual python environment and activate the environment
+ create an environment file to add the API keys to OPENAI and TAVILY with the following names : OPENAI_API_KEY and TAVILY_API_KEY
+ Open terminal and execute the following command to install required packages and to conainterise and run the application. And you can test your application from any API testing tool of your choice.
```console
pip install -r requirements.txt
docker compose up --build
```
## API's 
+ Thread Creation API : This is required for Checkpointers
+ Generate Article API: This will generate the Sports article for the given event
+ Edit Article API: This is for the Human in loop to interfere and edit the article if required
+ Confirm Article API: This is for the Human in the loop to confirm the article for publishing after evaluating and editing
+ Delete Thread API: This is to delete a particular thread from the database
+ Sessions API: This is to list all the active threads

 ## Below is an sample ARTICLE generated on the India vs New Zealand Champions Trophy 2025 Final Cricket Match
AI GENERATED ARTICLE:

**India triumphed over New Zealand by four wickets in the ICC Champions Trophy 2025 final held at the Dubai International Stadium on March 9, 2025. Chasing 252, India reached 254/6 in 49 overs, securing their third Champions Trophy title after victories in 2002 and 2013. Rohit Sharma led the chase with a stellar 76, forming a 105-run opening stand with Shubman Gill. Earlier, New Zealand posted 251/7, with Rachin Ravindra (37) and Will Young (15) adding 57 runs. Despite missed catches, India held their nerve to clinch a historic win in a tense finale.**
