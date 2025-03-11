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

 ## Below is a sample ARTICLE generated on the India vs New Zealand Champions Trophy 2025 Final Cricket Match
AI GENERATED ARTICLE:

**In the thrilling final of the 2025 ICC Champions Trophy, which was held at the iconic Dubai International Stadium, India emerged victorious against New Zealand by a margin of four wickets. The match showcased exceptional cricketing talent and strategic play from both teams. Captain Rohit Sharma was the standout performer for the Indian side, scoring a remarkable 76 runs off solid batting and showing composure under pressure. He formed a pivotal opening partnership with Shubman Gill, who contributed 31 runs to solidify their position at the start of the innings. This strong foundation set the tone for the chase, as they put on an impressive 105 runs together.**

**New Zealand, batting first, managed to post a total of 251 runs for the loss of 7 wickets, which presented a formidable challenge for the Indian batsmen. Rachin Ravindra played a crucial role in their innings, adding 37 runs, while Will Young contributed a modest 15 runs. The New Zealand batting lineup saw a mini-collapse during the match, which raised concerns, but they ultimately managed to set a competitive target for India nonetheless.**

**In response, the Indian team demonstrated their batting prowess and resilience, successfully chasing down the target and finishing their innings with 254 runs for the loss of 6 wickets in 49 overs. This triumphant victory not only secured India's place in cricketing history but also marked their third Champions Trophy title, having previously won the tournament in 2002 and 2013, further solidifying their reputation as one of the leading teams in international cricket.**
