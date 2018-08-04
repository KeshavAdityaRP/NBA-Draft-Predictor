from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd 
import time
import csv

def crawlPlayerLinks(rootUrl,year):
    bio = dict()
    parseFlag = 0
    count = 1
    try:
        # url that we are scraping
        url = rootUrl
        # this is the html from the given url
        html = urlopen(url)
        soup = BeautifulSoup(html, "html.parser")
        parseFlag = 1
    except:
        print ("Unable to Parse Draft Page")    

    if parseFlag == 1:
        results = soup.find('table', {"id" : "stats"}).find('tbody').find_all('tr')

        sampleSize = len(results)

        for result in results:
            if count <= sampleSize:
                flag = 0
                try:
                    result1 = result.find('td', {"data-stat" : "player"}).find('a')
                    result2 = result.find('td', {"data-stat" : "college_name"}) 
                    result3 = result.find('td', {"data-stat" : "pick_overall"})           
                    bio['link'] = result1["href"]
                    bio['name'] = result1.text
                    bio['year'] = year
                    bio['pick'] = result3["csk"]
                    if result2["csk"] == "Zzz":
                        bio["university"] = ""
                    else:
                        bio["university"] = result2["csk"] 
                        flag = 1  
                except:
                    pass 
                if flag == 1:
                    time.sleep(2)
                    crawlPlayerStats("http://www.basketball-reference.com" + bio['link'], bio)
            print (count," / ",sampleSize, " Completed")
            count = count + 1
                         
    
def crawlPlayerStats(playerUrl, bio):
    stats = dict()
    parseFlag = 0 
    crawled = 0
    try:    
        # url that we are scraping
        url = playerUrl
        # this is the html from the given url
        html = urlopen(url)
        soup = BeautifulSoup(html, "html.parser")
        parseFlag = 1
    except:
        print ("Unable to Parse Player Page")  
        print (bio)  

    if parseFlag == 1:
        try:
            result = soup.find('div', { "id" : "wrap" }).find('div',  {"id" : "content"}).find('div',  {"id" : "all_all_college_stats"})
            result = str(result)
            result = result.replace("<!--","").replace("-->","")
            soup = BeautifulSoup(result, "html.parser")
            results = soup.find('div', {"class" : "table_outer_container"}).find('div', {"id":"div_all_college_stats"}).find('table').find('tfoot').find('tr').findAll('td', {"class" : "right"})
            stats_length = len(results)
            if stats_length == 23:
                for result in results:
                    stats[result["data-stat"]] = result.text
                crawled = 1    
            else:
                print ("Unknown number of stats") 
                print (bio)       
            if crawled == 1:
                print ("Sucessful Crawl ...")
                bio_summary = []
                stats_summary = []
                bio_summary.extend((bio['pick'],bio['year'],bio['name'],bio["university"],bio['link'])) 
                stats_summary.extend((stats['g'],stats['mp'],stats['fg'],stats['fga'],stats['fg3'],stats['fg3a'],stats['ft'],stats['fta'],stats['orb'],stats['trb'],stats['ast'],stats['stl'],stats['blk'],stats['tov'],stats['pf'],stats['pts'],stats['fg_pct'],stats['fg3_pct'],stats['ft_pct'],stats['mp_per_g'],stats['pts_per_g'],stats['trb_per_g'],stats['ast_per_g']))
                bio_summary_str = ','.join(map(str, bio_summary))
                stats_summary_str = ','.join(map(str, stats_summary)) 
                summary =  bio_summary_str + "," + stats_summary_str
                # bio_summary = ",".join(bio.values())
                # stats_summary = ",".join(stats.values())
                # summary = bio_summary + ","  + stats_summary
                # bio_key = ','.join(bio.keys())
                # stats_key = ','.join(stats.keys())
                # summary_key = bio_key + ","  + stats_key                
                storeCsv(summary, bio["year"])

        except:
            print ("College Stats not Found")  
            print (bio)            


def storeCsv(summary, year):
    filename = "Datasets/" + year + "Draft.csv"
    with open(filename,'a') as file:
        file.write(summary)
        file.write('\n') 
    print ("Stored Sucessfully")       

for i in range(2003,2004):
    print ("Initiated Crawler for :", str(i))
    year = str(i)
    rootUrl = "https://www.basketball-reference.com/draft/NBA_" + year + ".html"
    crawlPlayerLinks(rootUrl, year)
