"""
Fuzzy Semantic Encoder - Optimized Version
==========================================

Tối ưu hóa dựa trên pipeline chính:
- Membership functions theo ICC paper (giống fuzzy_system.py)
- 8 rules Mamdani theo Table II
- Fallback mechanism với confidence thresholds
- Tương thích với scikit-fuzzy và manual min-max
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


def _trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function"""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


# Confidence thresholds (giống pipeline chính)
CONFIDENCE_OVERRIDE_THRESHOLD = 0.29
CLASS_CONFIDENCE_THRESHOLDS = {
    "nutrient_deficiency": 0.90,
    "fungal_risk": 0.80,
}

SEMANTIC_CLASSES = [
    "optimal",
    "nutrient_deficiency",
    "fungal_risk",
    "water_deficit_acidic",
    "water_deficit_alkaline",
    "acidic_soil",
    "alkaline_soil",
    "heat_stress",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(SEMANTIC_CLASSES)}


class FuzzySemanticEncoderOptimized:
    """
    Fuzzy Semantic Encoder tối ưu - dùng membership và rules giống pipeline chính
    """
    
    def __init__(self, use_scikit_fuzzy: bool = False):
        """
        Parameters:
        -----------
        use_scikit_fuzzy : bool
            Nếu True, dùng scikit-fuzzy simulator. Nếu False, dùng manual min-max (giống pipeline chính)
        """
        self.use_scikit_fuzzy = use_scikit_fuzzy
        self.semantic_classes = {
            i: {
                'name': name,
                'description': self._get_description(name),
                'action': self._get_action(name),
                'priority': self._get_priority(name)
            }
            for i, name in enumerate(SEMANTIC_CLASSES)
        }
        
        if use_scikit_fuzzy:
            self._setup_fuzzy_system()
    
    def _get_description(self, class_name: str) -> str:
        descriptions = {
            'optimal': 'Optimal growing conditions',
            'nutrient_deficiency': 'Low nitrogen levels with acidic soil',
            'fungal_risk': 'High humidity and cool temperature',
            'water_deficit_acidic': 'Low moisture with acidic soil',
            'water_deficit_alkaline': 'Low moisture with alkaline soil',
            'acidic_soil': 'Soil pH too low',
            'alkaline_soil': 'Soil pH too high',
            'heat_stress': 'High temperature with low humidity',
        }
        return descriptions.get(class_name, 'Unknown condition')
    
    def _get_action(self, class_name: str) -> str:
        actions = {
            'optimal': 'Monitor',
            'nutrient_deficiency': 'Apply Fertilizer',
            'fungal_risk': 'Apply Fungicide + Ventilate',
            'water_deficit_acidic': 'Irrigate + Apply Lime',
            'water_deficit_alkaline': 'Irrigate + Apply Sulfur',
            'acidic_soil': 'Apply Lime',
            'alkaline_soil': 'Apply Sulfur',
            'heat_stress': 'Increase Irrigation + Shade',
        }
        return actions.get(class_name, 'Monitor')
    
    def _get_priority(self, class_name: str) -> str:
        priorities = {
            'optimal': 'Low',
            'nutrient_deficiency': 'High',
            'fungal_risk': 'High',
            'water_deficit_acidic': 'Critical',
            'water_deficit_alkaline': 'Critical',
            'acidic_soil': 'Medium',
            'alkaline_soil': 'Medium',
            'heat_stress': 'High',
        }
        return priorities.get(class_name, 'Low')
    
    def _setup_fuzzy_system(self):
        """Setup scikit-fuzzy system (nếu dùng use_scikit_fuzzy=True)"""
        # Membership functions theo ICC paper (giống pipeline chính)
        # Moisture
        self.moisture = ctrl.Antecedent(np.arange(0, 101, 1), 'moisture')
        self.moisture['dry'] = fuzz.trimf(self.moisture.universe, [15, 20, 30])
        self.moisture['ideal'] = fuzz.trimf(self.moisture.universe, [30, 45, 60])
        self.moisture['wet'] = fuzz.trimf(self.moisture.universe, [60, 70, 85])
        
        # pH
        self.ph = ctrl.Antecedent(np.arange(4.0, 9.1, 0.1), 'ph')
        self.ph['acidic'] = fuzz.trimf(self.ph.universe, [4.5, 5.0, 5.8])
        self.ph['ideal'] = fuzz.trimf(self.ph.universe, [6.0, 6.3, 6.8])
        self.ph['alkaline'] = fuzz.trimf(self.ph.universe, [6.8, 7.5, 8.5])
        
        # Nitrogen
        self.nitrogen = ctrl.Antecedent(np.arange(0, 251, 1), 'nitrogen')
        self.nitrogen['low'] = fuzz.trimf(self.nitrogen.universe, [0, 20, 40])
        self.nitrogen['adequate'] = fuzz.trimf(self.nitrogen.universe, [50, 80, 100])
        self.nitrogen['high'] = fuzz.trimf(self.nitrogen.universe, [150, 200, 250])
        
        # Temperature
        self.temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
        self.temperature['cool'] = fuzz.trimf(self.temperature.universe, [15, 18, 22])
        self.temperature['ideal'] = fuzz.trimf(self.temperature.universe, [22, 24, 26])
        self.temperature['hot'] = fuzz.trimf(self.temperature.universe, [26, 30, 38])
        
        # Humidity
        self.humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
        self.humidity['dry'] = fuzz.trimf(self.humidity.universe, [40, 50, 60])
        self.humidity['ideal'] = fuzz.trimf(self.humidity.universe, [60, 65, 70])
        self.humidity['humid'] = fuzz.trimf(self.humidity.universe, [70, 80, 90])
        
        # Output
        self.semantic_class = ctrl.Consequent(np.arange(0, 8, 1), 'semantic_class')
        for i in range(8):
            self.semantic_class[f'class_{i}'] = fuzz.trimf(
                self.semantic_class.universe,
                [max(0, i-0.5), i, min(7, i+0.5)]
            )
        
        # Rules theo Table II (giống pipeline chính)
        self.rules = []
        self.rules.append(ctrl.Rule(
            self.moisture['ideal'] & self.ph['ideal'] & self.nitrogen['adequate'] &
            self.temperature['ideal'] & self.humidity['ideal'],
            self.semantic_class['class_0']
        ))
        self.rules.append(ctrl.Rule(
            self.nitrogen['low'] & self.ph['acidic'],
            self.semantic_class['class_1']
        ))
        self.rules.append(ctrl.Rule(
            self.humidity['humid'] & self.temperature['cool'],
            self.semantic_class['class_2']
        ))
        self.rules.append(ctrl.Rule(
            self.moisture['dry'] & self.ph['acidic'],
            self.semantic_class['class_3']
        ))
        self.rules.append(ctrl.Rule(
            self.moisture['dry'] & self.ph['alkaline'],
            self.semantic_class['class_4']
        ))
        self.rules.append(ctrl.Rule(
            self.ph['acidic'] & (self.moisture['ideal'] | self.moisture['wet']),
            self.semantic_class['class_5']
        ))
        self.rules.append(ctrl.Rule(
            self.ph['alkaline'] & (self.moisture['ideal'] | self.moisture['wet']),
            self.semantic_class['class_6']
        ))
        self.rules.append(ctrl.Rule(
            self.temperature['hot'] & self.humidity['dry'],
            self.semantic_class['class_7']
        ))
        
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)
    
    def _calculate_memberships_icc(self, moisture: float, ph: float, nitrogen: float,
                                   temperature: float, humidity: float) -> Dict:
        """Tính membership theo ICC paper (giống pipeline chính)"""
        return {
            'moisture': {
                'dry': _trimf(moisture, 15, 20, 30),
                'ideal': _trimf(moisture, 30, 45, 60),
                'wet': _trimf(moisture, 60, 70, 85),
            },
            'ph': {
                'acidic': _trimf(ph, 4.5, 5.0, 5.8),
                'ideal': _trimf(ph, 6.0, 6.3, 6.8),
                'alkaline': _trimf(ph, 6.8, 7.5, 8.5),
            },
            'nitrogen': {
                'low': _trimf(nitrogen, 0, 20, 40),
                'adequate': _trimf(nitrogen, 50, 80, 100),
                'high': _trimf(nitrogen, 150, 200, 250),
            },
            'temperature': {
                'cool': _trimf(temperature, 15, 18, 22),
                'ideal': _trimf(temperature, 22, 24, 26),
                'hot': _trimf(temperature, 26, 30, 38),
            },
            'humidity': {
                'dry': _trimf(humidity, 40, 50, 60),
                'ideal': _trimf(humidity, 60, 65, 70),
                'humid': _trimf(humidity, 70, 80, 90),
            },
        }
    
    def _calculate_rule_strengths(self, memberships: Dict) -> Dict[str, float]:
        """Tính rule strengths theo Table II (giống pipeline chính)"""
        strengths = {name: 0.0 for name in SEMANTIC_CLASSES}
        m = memberships
        
        moist_ge_30 = max(m['moisture']['ideal'], m['moisture']['wet'])
        
        strengths['water_deficit_acidic'] = min(m['moisture']['dry'], m['ph']['acidic'])
        strengths['water_deficit_alkaline'] = min(m['moisture']['dry'], m['ph']['alkaline'])
        strengths['acidic_soil'] = min(m['ph']['acidic'], moist_ge_30)
        strengths['alkaline_soil'] = min(m['ph']['alkaline'], moist_ge_30)
        strengths['optimal'] = min(
            m['moisture']['ideal'],
            m['ph']['ideal'],
            m['nitrogen']['adequate'],
            m['temperature']['ideal'],
            m['humidity']['ideal'],
        )
        strengths['heat_stress'] = min(m['temperature']['hot'], m['humidity']['dry'])
        strengths['nutrient_deficiency'] = min(m['nitrogen']['low'], m['ph']['acidic'])
        strengths['fungal_risk'] = min(m['humidity']['humid'], m['temperature']['cool'])
        
        return strengths
    
    def _crisp_fallback(self, moisture: float, ph: float, nitrogen: float,
                        temperature: float, humidity: float, ndi_label: str | None = None,
                        pdi_label: str | None = None) -> str:
        """Fallback mechanism (giống pipeline chính)"""
        if (30 <= moisture <= 60 and 6.0 <= ph <= 6.8 and 50 <= nitrogen <= 100 and
            22 <= temperature <= 26 and 60 <= humidity <= 70):
            return "optimal"
        if moisture < 30 and ph < 5.8:
            return "water_deficit_acidic"
        if moisture < 30 and ph > 7.5:
            return "water_deficit_alkaline"
        if ph < 5.8 and moisture >= 30:
            return "acidic_soil"
        if ph > 7.5 and moisture >= 30:
            return "alkaline_soil"
        if temperature > 30 and humidity < 60:
            return "heat_stress"
        if humidity > 80 and temperature < 22:
            return "fungal_risk"
        if nitrogen < 40 and ph < 5.8:
            return "nutrient_deficiency"
        if ndi_label == "High":
            return "nutrient_deficiency"
        if pdi_label == "High":
            return "fungal_risk"
        return "optimal"
    
    def encode(self, moisture: float, ph: float, nitrogen: float,
               temperature: float, humidity: float, ndi_label: str | None = None,
               pdi_label: str | None = None) -> Tuple[int, float, Dict]:
        """
        Encode sensor data into semantic class
        
        Returns:
        --------
        semantic_class : int
            Semantic class ID (0-7)
        confidence : float
            Confidence score (0-1)
        memberships : dict
            Membership degrees
        """
        memberships = self._calculate_memberships_icc(moisture, ph, nitrogen, temperature, humidity)
        
        if self.use_scikit_fuzzy:
            try:
                self.simulator.input['moisture'] = moisture
                self.simulator.input['ph'] = ph
                self.simulator.input['nitrogen'] = nitrogen
                self.simulator.input['temperature'] = temperature
                self.simulator.input['humidity'] = humidity
                self.simulator.compute()
                raw_output = self.simulator.output['semantic_class']
                semantic_class = int(np.round(raw_output))
                semantic_class = max(0, min(7, semantic_class))
                # Tính confidence từ rule strengths
                strengths = self._calculate_rule_strengths(memberships)
                confidence = strengths[SEMANTIC_CLASSES[semantic_class]]
            except:
                crisp_class = self._crisp_fallback(moisture, ph, nitrogen, temperature, humidity, ndi_label, pdi_label)
                semantic_class = CLASS_TO_ID[crisp_class]
                confidence = 0.0
        else:
            # Manual min-max (giống pipeline chính)
            strengths = self._calculate_rule_strengths(memberships)
            
            # Bổ sung heuristic theo Step 4 (NDI/PDI) nếu có (giống pipeline chính)
            if ndi_label == "High":
                strengths["nutrient_deficiency"] = max(strengths["nutrient_deficiency"], 1.0)
            if pdi_label == "High" and humidity > 70 and temperature < 22:
                strengths["fungal_risk"] = max(strengths["fungal_risk"], 1.0)
            
            best_class = max(SEMANTIC_CLASSES, key=lambda name: (strengths[name], -CLASS_TO_ID[name]))
            raw_confidence = strengths[best_class]
            
            # Fallback mechanism với confidence thresholds (giống pipeline chính)
            crisp_class = self._crisp_fallback(moisture, ph, nitrogen, temperature, humidity, ndi_label, pdi_label)
            threshold = CLASS_CONFIDENCE_THRESHOLDS.get(best_class, CONFIDENCE_OVERRIDE_THRESHOLD)
            
            if raw_confidence == 0.0:
                best_class = crisp_class
                confidence = 0.0
            elif raw_confidence < threshold:
                best_class = crisp_class
                confidence = max(raw_confidence, 1.0 if crisp_class == "optimal" else 0.7)
            else:
                confidence = raw_confidence
            
            semantic_class = CLASS_TO_ID[best_class]
        
        return semantic_class, float(confidence), memberships
    
    def decode(self, semantic_class: int) -> Dict:
        """Decode semantic class to human-readable information"""
        if semantic_class not in self.semantic_classes:
            raise ValueError(f"Invalid semantic class: {semantic_class}")
        return self.semantic_classes[semantic_class]
    
    def encode_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode a batch of sensor data"""
        results = []
        for idx, row in df.iterrows():
            semantic_class, confidence, _ = self.encode(
                row['moisture'],
                row['ph'],
                row['nitrogen'],
                row['temperature'],
                row['humidity'],
                row.get('ndi_label'),
                row.get('pdi_label'),
            )
            results.append({
                'semantic_class': semantic_class,
                'confidence': confidence,
                'class_name': self.semantic_classes[semantic_class]['name'],
                'action': self.semantic_classes[semantic_class]['action'],
                'priority': self.semantic_classes[semantic_class]['priority']
            })
        result_df = df.copy()
        result_df = pd.concat([result_df, pd.DataFrame(results)], axis=1)
        return result_df

