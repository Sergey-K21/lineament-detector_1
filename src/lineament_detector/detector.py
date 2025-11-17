
import numpy as np
import rasterio
from rasterio.plot import show
from skimage import filters, feature, morphology, measure
from skimage.exposure import equalize_hist
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from rasterio.transform import Affine


class Lineament_Detector:
    def __init__(self, min_length_pixels: int = 300,
                 canny_sigma: float = 0.8,
                 simplify_tolerance: float = 2.0):
        """
        Полнофункциональный детектор линеаментов

        Параметры:
        min_length_pixels - минимальная длина в пикселях
        canny_sigma - параметр размытия для детектора границ
        simplify_tolerance - степень упрощения линий
        """
        self.min_length = min_length_pixels
        self.canny_sigma = canny_sigma
        self.simplify_tol = simplify_tolerance
        self.transform = None
        self.crs = None

    def _preprocess_dem(self, dem: np.ndarray) -> np.ndarray:
        """Предварительная обработка ЦМР"""
        try:
            # Нормализация гистограммы
            dem_norm = equalize_hist(dem)
            # Медианная фильтрация
            dem_filtered = filters.median(dem_norm, morphology.disk(3))
            # Гауссово размытие
            return filters.gaussian(dem_filtered, sigma=1.0)
        except Exception as e:
            print(f"Ошибка предобработки: {e}")
            return dem

    def _detect_edges(self, dem: np.ndarray) -> np.ndarray:
        """Детектирование границ с улучшенным алгоритмом"""
        try:
            # Расчет градиентов
            dy, dx = np.gradient(dem)
            gradient_mag = np.sqrt(dx**2 + dy**2)

            # Адаптивный детектор Canny
            p_low = np.percentile(gradient_mag, 25)
            p_high = np.percentile(gradient_mag, 75)

            edges = feature.canny(
                gradient_mag,
                sigma=self.canny_sigma,
                low_threshold=p_low,
                high_threshold=p_high
            )

            # Улучшение связности линий
            return morphology.binary_closing(edges, morphology.disk(2))
        except Exception as e:
            print(f"Ошибка детектирования границ: {e}")
            return np.zeros_like(dem, dtype=bool)

    def _vectorize_edges(self, edges: np.ndarray) -> gpd.GeoDataFrame:
        try:
            from skimage.transform import probabilistic_hough_line

    # Параметры для детекции прямых
            hough_params = {
            'threshold': 50,
            'line_length': self.min_length,
            'line_gap': 5,
            'theta': np.linspace(-np.pi/2, np.pi/2, 180)  # Шаг 1 градус
            }

    # Детекция прямых линий методом Хафа
            lines = probabilistic_hough_line(edges, **hough_params)

            features = []
            for line in lines:
                p0, p1 = line
                if self.transform:
            # Преобразование координат
                    x0, y0 = self.transform * (p0[0], p0[1])
                    x1, y1 = self.transform * (p1[0], p1[1])
                    line_geom = LineString([(x0, y0), (x1, y1)])
                else:
                    line_geom = LineString([(p0[0], p0[1]), (p1[0], p1[1])])

        # Фильтр по минимальной длине
                if line_geom.length >= self.min_length:
                    features.append(line_geom)

            return gpd.GeoDataFrame(geometry=features, crs=self.crs)

        except Exception as e:
            print(f"Ошибка векторизации: {e}")
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

    def _postprocess_lines(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

        if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

        try:
        # Объединение линий с преобразованием MultiLineString в список LineString
            merged = linemerge(gdf.geometry.tolist())  # Преобразуем в список геометрий

            processed = []

        # Обработка как единой линии или набора линий
            if merged.geom_type == 'MultiLineString':
                for line in merged.geoms:  # Используем .geoms для MultiLineString
                    try:
                        simplified = line.simplify(self.simplify_tol)
                        if not simplified.is_empty:
                            min_length = self.min_length * abs(self.transform.a) if self.transform else self.min_length
                            if simplified.length >= min_length:
                                processed.append(simplified)
                    except Exception as e:
                        print(f"Ошибка упрощения линии: {e}")
                        continue
            elif merged.geom_type == 'LineString':
                try:
                    simplified = merged.simplify(self.simplify_tol)
                    if not simplified.is_empty:
                        min_length = self.min_length * abs(self.transform.a) if self.transform else self.min_length
                        if simplified.length >= min_length:
                            processed.append(simplified)
                except Exception as e:
                    print(f"Ошибка упрощения линии: {e}")
            else:
                print(f"Неожиданный тип геометрии: {merged.geom_type}")

            return gpd.GeoDataFrame(geometry=processed, crs=self.crs)

        except Exception as e:
            print(f"Ошибка постобработки: {e}")
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

    def _calculate_stats(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Расчет статистики"""
        if not isinstance(gdf, gpd.GeoDataFrame) or gdf.empty:
            return {
                'total_count': 0,
                'total_length': 0.0,
                'avg_length': 0.0,
                'max_length': 0.0,
                'orientation_bins': (np.array([]), np.array([]))
            }

        try:
            lengths = []
            orientations = []

            for geom in gdf.geometry:
                if not geom.is_empty:
                    lengths.append(geom.length)
                    coords = np.array(geom.coords)
                    if len(coords) >= 2:
                        dx = coords[-1,0] - coords[0,0]
                        dy = coords[-1,1] - coords[0,1]
                        angle = np.degrees(np.arctan2(dy, dx)) % 180
                        orientations.append(angle)

            if not lengths:
                return {
                    'total_count': 0,
                    'total_length': 0.0,
                    'avg_length': 0.0,
                    'max_length': 0.0,
                    'orientation_bins': (np.array([]), np.array([]))
                }

            return {
                'total_count': len(lengths),
                'total_length': float(np.sum(lengths)),
                'avg_length': float(np.mean(lengths)),
                'max_length': float(np.max(lengths)),
                'orientation_bins': np.histogram(orientations, bins=np.linspace(0, 180, 13)) if orientations else (np.array([]), np.array([]))
            }

        except Exception as e:
            print(f"Ошибка расчета статистики: {e}")
            return {
                'total_count': 0,
                'total_length': 0.0,
                'avg_length': 0.0,
                'max_length': 0.0,
                'orientation_bins': (np.array([]), np.array([]))
            }

    def process(self, input_raster: str,
               output_shp: str,
               output_png: str) -> Tuple[Optional[gpd.GeoDataFrame], Dict]:
        """
        Основной метод обработки

        Параметры:
        input_raster - путь к входному растру
        output_shp - путь для сохранения шейп-файла
        output_png - путь для сохранения визуализации

        Возвращает:
        Кортеж (GeoDataFrame с линеаментами, статистика)
        """
        try:
            # Загрузка данных
            dem = dem

            # Обработка данных
            processed_dem = self._preprocess_dem(dem)
            edges = self._detect_edges(processed_dem)
            raw_lines = self._vectorize_edges(edges)
            final_lines = self._postprocess_lines(raw_lines)
            stats = self._calculate_stats(final_lines)

            # Сохранение результатов
            if not final_lines.empty:
                try:
                    final_lines.to_file(output_shp)
                except Exception as e:
                    print(f"Ошибка сохранения шейп-файла: {e}")

            # Визуализация
            try:
                fig, ax = plt.subplots(figsize=(12, 12))
                show(dem, ax=ax, cmap='terrain')

                if not final_lines.empty:
                    final_lines.plot(ax=ax, color='red', linewidth=1.5)

                ax.set_title(f"Линеаменты (N={stats['total_count']}, L={stats['total_length']:.1f} м)")
                plt.savefig(output_png, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Ошибка визуализации: {e}")
                plt.close('all')

            return final_lines, stats

        except Exception as e:
            print(f"Критическая ошибка обработки: {e}")
            return None, {
                'total_count': 0,
                'total_length': 0.0,
                'avg_length': 0.0,
                'max_length': 0.0,
                'orientation_bins': (np.array([]), np.array([]))
            }
