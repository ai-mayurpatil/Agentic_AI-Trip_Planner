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

class TripTasks:
    def __init__(self):
        pass
    
    def city_selection_task(self, agent, inputs):
        return Task(
            name="city_selection",  # Set task name for later extraction
            description=(
                f"Analyze user preferences and select best destinations:\n"
                f"- Travel Type: {inputs['travel_type']}\n"
                f"- Interests: {inputs['interests']}\n"
                f"- Season: {inputs['season']}\n"
                "Output: Provide 3 city options with a brief rationale for each."
            ),
            agent=agent,
            expected_output="Bullet-point list of 3 cities with 2-sentence explanations each."
        )
    
    def city_research_task(self, agent, city):
        return Task(
            name="city_research",  # Set task name
            description=(
                f"Provide detailed insights about {city} including:\n"
                "- Top 5 attractions\n"
                "- Local cuisine highlights\n"
                "- Cultural norms/etiquette\n"
                "- Recommended accommodation areas\n"
                "- Transportation tips"
            ),
            agent=agent,
            expected_output="Organized sections with clear headings and bullet points."
        )
    
    def itinerary_creation_task(self, agent, inputs, city):
        return Task(
            name="itinerary",  # Set task name
            description=(
                f"Create a {inputs['duration']}-day itinerary for {city} including:\n"
                "- Daily schedule with time allocations\n"
                "- Activity sequencing\n"
                "- Transportation between locations\n"
                "- Meal planning suggestions"
            ),
            agent=agent,
            expected_output="Day-by-day table format with time slots and activity details."
        )
    
    def budget_planning_task(self, agent, inputs, itinerary):
        return Task(
            name="budget",  # Set task name
            description=(
                f"Create a budget plan for the selected budget range ({inputs['budget']}) covering:\n"
                "- Accommodation costs\n"
                "- Transportation expenses\n"
                "- Activity fees\n"
                "- Meal budget\n"
                "- Emergency funds allocation"
            ),
            agent=agent,
            context=[itinerary],
            expected_output="Itemized budget table with total cost analysis."
        )
