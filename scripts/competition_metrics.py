#!/usr/bin/env python3
"""
공모전 평가 메트릭 구현
- AP (Average Precision): 50%
- WLL (Weighted LogLoss): 50%
- Final Score: 0.5 * AP + 0.5 * (1 / (1 + WLL))
"""

import numpy as np
from sklearn.metrics import average_precision_score, log_loss

def weighted_log_loss(y_true, y_pred, class_weights=None):
    """
    Weighted LogLoss 계산
    클릭(1)과 비클릭(0)의 기여도를 50:50으로 맞춤
    """
    if class_weights is None:
        # 클래스 비율에 따른 가중치 계산
        pos_ratio = np.mean(y_true)
        neg_ratio = 1 - pos_ratio
        
        # 50:50 비율로 가중치 조정
        pos_weight = 0.5 / pos_ratio if pos_ratio > 0 else 0
        neg_weight = 0.5 / neg_ratio if neg_ratio > 0 else 0
        
        # 샘플별 가중치
        sample_weights = np.where(y_true == 1, pos_weight, neg_weight)
    else:
        sample_weights = np.where(y_true == 1, class_weights[1], class_weights[0])
    
    # 안정성을 위한 클리핑
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # 가중 로그 손실 계산
    wll = log_loss(y_true, y_pred_clipped, sample_weight=sample_weights)
    
    return wll

def competition_score(y_true, y_pred):
    """
    공모전 평가 점수 계산
    Score = 0.5 * AP + 0.5 * (1 / (1 + WLL))
    """
    # Average Precision 계산
    ap = average_precision_score(y_true, y_pred)
    
    # Weighted LogLoss 계산
    wll = weighted_log_loss(y_true, y_pred)
    
    # 최종 점수
    wll_component = 1 / (1 + wll)
    final_score = 0.5 * ap + 0.5 * wll_component
    
    return {
        'AP': ap,
        'WLL': wll,
        'WLL_Component': wll_component,
        'Final_Score': final_score
    }

def evaluate_predictions(y_true, y_pred, verbose=True):
    """
    예측 결과를 공모전 방식으로 평가
    """
    results = competition_score(y_true, y_pred)
    
    if verbose:
        print(f"🏆 공모전 평가 결과:")
        print(f"   📊 AP (Average Precision): {results['AP']:.6f}")
        print(f"   📉 WLL (Weighted LogLoss): {results['WLL']:.6f}")
        print(f"   🔢 WLL Component (1/(1+WLL)): {results['WLL_Component']:.6f}")
        print(f"   🎯 Final Score: {results['Final_Score']:.6f}")
        print(f"")
        print(f"💡 점수 구성:")
        print(f"   AP 기여분 (50%): {0.5 * results['AP']:.6f}")
        print(f"   WLL 기여분 (50%): {0.5 * results['WLL_Component']:.6f}")
    
    return results

if __name__ == "__main__":
    # 테스트 예시
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.1, 1000)  # 10% 클릭률
    y_pred = np.random.beta(1, 9, 1000)  # 낮은 확률 예측
    
    print("🧪 테스트 예시:")
    evaluate_predictions(y_true, y_pred)
