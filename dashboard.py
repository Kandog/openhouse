"""Dashboard module for visitor statistics."""

from datetime import datetime
from collections import defaultdict
import database


class VisitorDashboard:
    def __init__(self):
        self.daily_stats = defaultdict(list)
        self.hourly_stats = defaultdict(lambda: defaultdict(list))
    
    def record_visitor(self, visitor_id: int, name: str, visit_type: str = "new"):
        """Record a visitor interaction."""
        now = datetime.now()
        date_key = now.strftime("%Y-%m-%d")
        hour_key = now.strftime("%H:00")
        
        self.daily_stats[date_key].append({
            "id": visitor_id,
            "name": name,
            "time": now.isoformat(),
            "type": visit_type,
        })
        
        self.hourly_stats[date_key][hour_key].append({
            "id": visitor_id,
            "name": name,
            "time": now.isoformat(),
            "type": visit_type,
        })
    
    def get_daily_stats(self, date: str = None):
        """Get stats for a specific day or today."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        visitors = self.daily_stats.get(date, [])
        return {
            "date": date,
            "total_visits": len(visitors),
            "unique_visitors": len(set(v["id"] for v in visitors)),
            "visitors": visitors,
        }
    
    def get_hourly_stats(self, date: str = None):
        """Get hourly breakdown for a specific day."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        hourly = self.hourly_stats.get(date, {})
        result = []
        
        for hour in sorted(hourly.keys()):
            visitors = hourly[hour]
            result.append({
                "hour": hour,
                "count": len(visitors),
                "visitors": visitors,
            })
        
        return result
