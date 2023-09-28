MAIN_PATH = "/home/ec2-user/personalizing-llm-hl"
DATA_PATH = MAIN_PATH + "/data"

INTERESTS = {
   "tech enthusiast":{
      "description": "Passionate about all things tech, from gadgets to software.",
      "speaking_style": "Shares tech news, insights, and engages in technical discussions using industry-specific terms."
   },
   "foodie": {
      "description": "Obsessed with food, culinary experiences, and restaurants.",
      "speaking_style": "Uses sensory language, shares food reviews, and recommends restaurants with mouthwatering descriptions."
   },
   "travel explorer": {
      "description": "A globe-trotter who seeks adventure in travel and exploration.",
      "speaking_style": "Shares travel tips, stories, and breathtaking photos from various destinations."     
   },
   "fashionista": {
      "description": "A fashion trendsetter who lives for style and aesthetics.",
      "speaking_style": "Shares outfit photos, fashion tips, and critiques popular fashion trends."
   },
   "gamer": {
      "description": "Devoted to video games and gaming culture.",
      "speaking_style": "Discusses game strategies, reviews, and engages in gaming-related banter."
   },
   "bookworm": {
      "description": "Avid reader with a passion for literature and storytelling.",
      "speaking_style": "Shares book recommendations, reviews, and favorite literary quotes."
   },
   "fitness guru": {
      "description": "Fitness enthusiast focused on health, exercise, and nutrition.",
      "speaking_style": "Shares workout routines, nutrition tips, and progress updates."
   },
   "film buff": {
      "description": "A cinephile who loves movies and all things related to cinema.",
      "speaking_style": "Reviews films, discusses directors, and participates in film debates."  
   },
   "celebrity gossipmonger": {
      "description": "They stay up-to-date with the latest celebrity gossip and news. They express strong emotions from admiration for their favorite celebrities to criticism when controversies arise.",
      "speaking_style": "Opinionated and engages in debates with other users about celebrity drama, fashion choices, and personal lives." 
   },
   "comedian": {
      "description": "A humorist dedicated to making people laugh.",
      "speaking_style": "Shares jokes, funny anecdotes, and shows where they are performing."    
   },
   "political commentator": {
      "description": "Engages in political discussions and shares opinions on current events.",
      "speaking_style": "Analyzes political developments, shares news articles, and interacts with followers on pressing issues."    
   },
   "parenting blogger": {
      "description": "Provides parenting advice, tips, and insights into family life.",
      "speaking_style": "Shares parenting experiences, offers guidance, and connects with other parents."         
   },
   "health & wellness influencer": {
      "description": "Promotes a healthy lifestyle, fitness, and self-care.",
      "speaking_style": "Shares wellness tips, mindfulness techniques, and personal health journeys."         
   }, 
   "sports fanatic": {
      "description": "Is deeply passionate about sports, often identified by their team apparel, encyclopedic sports knowledge, and unwavering team loyalty.",
      "speaking_style": "Speaks with intense enthusiasm, frequently tweets about results of the game, and emotional highs and lows tied to their team's performance."         
   }, 
   "art lover": {
      "description": "Enthusiastic about visual arts, from painting to sculpture.",
      "speaking_style": "Shares art discoveries, favorite artworks, art museums, and artistic inspirations."        
   },
   "science enthusiast": {
      "description": "Fascinated by scientific discoveries and innovations.",
      "speaking_style": "Shares scientific news, interesting facts, and engages in discussions about various fields of science."    
   },
   "music aficionado": {
      "description": "Passionate about music, concerts, and musical talent.",
      "speaking_style": "Shares music recommendations, reviews albums, and discusses favorite bands and artists."         
   },
   "humanitarian": {
      "description": "Dedicated to social causes and global humanitarian efforts.",
      "speaking_style": "Advocates for charitable organizations, shares stories of impact, and raises awareness about important issues."         
   },
   "history buff": {
      "description": "Enthusiastic about history, historical events, and cultures.",
      "speaking_style": "Shares historical facts, recommends history books, and engages in historical discussions."        
   },
   "professor": {
      "description": "Shares updates about their PhD students, conference talks, awards, and research from their laboratory.",
      "speaking_style": "Informative and scholarly tone, providing insights into their students' achievements, recent research findings, and academic milestones."         
   }
}

INTEREST_TYPES = ["tech enthusiast", "foodie", "travel explorer", "fashionista", "gamer", \
                  "bookworm", "fitness guru", "film buff", "celebrity gossipmonger", "comedian", \
                  "political commentator", "parenting blogger", "health & wellness influencer", "sports fanatic", "art lover", \
                  "science enthusiast", "music aficionado", "humanitarian", "history buff", "professor"
                  ]

# for classifier
INTEREST_IDXS = {value: index for index, value in enumerate(INTEREST_TYPES)}