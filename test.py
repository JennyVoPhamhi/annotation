from annotation import Annotation


new_anno = Annotation('BeadAnnotation_20180413.json')
#print(new_anno.getTurker('AAWQM8QY54NV'))
#print(new_anno.getCoords('AAWQM8QY54NV'))
print(new_anno.get_avg_time_per_click('AAWQM8QY54NV'))