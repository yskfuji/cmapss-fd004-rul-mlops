from .backtest import build_backtest_router
from .forecast import build_forecast_router
from .jobs import build_jobs_router
from .monitoring import build_monitoring_router
from .train import build_train_router

__all__ = [
	"build_backtest_router",
	"build_forecast_router",
	"build_jobs_router",
	"build_monitoring_router",
	"build_train_router",
]