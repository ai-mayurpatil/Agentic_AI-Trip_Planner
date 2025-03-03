from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

class TripAgents:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
    def city_selector_agent(self):
        return Agent(
            role='City Selection Expert',
            goal='Identify best cities to visit based on user preferences',
            backstory=(
                "An expert travel geographer with extensive knowledge about world cities "
                "and their cultural, historical, and entertainment offerings"
            ),
            llm=self.llm,
            verbose=True
        )
    
    def local_expert_agent(self):
        return Agent(
            role='Local Destination Expert',
            goal="Provide detailed insights about selected cities including top attractions, local customs, and hidden gems",
            backstory="A knowledgeable local guide with first-hand experience of the city's culture and attractions",
            llm=self.llm,
            verbose=True
        )
    
    def travel_planner_agent(self):
        return Agent(
            role='Professional Travel Planner',
            goal="Create detailed day-by-day itineraries with time allocations, transportation options, and activity sequencing",
            backstory="An experienced travel coordinator with perfect logistical planning skills",
            llm=self.llm,
            verbose=True
        )
    
    def budget_manager_agent(self):
        return Agent(
            role='Travel Budget Specialist',
            goal="Optimize travel plans to stay within budget while maximizing experience quality",
            backstory="A financial planner specializing in travel budgets and cost optimization",
            llm=self.llm,
            verbose=True
        )
