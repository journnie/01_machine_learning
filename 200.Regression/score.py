def evaluate_score(reg, X_test, y_test):
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score

    # 평가(검증) 데이터로 예측 수행 -> 예측 결과 y_pred구하기
    y_pred = reg.predict(X_test)

    # MSE
    mse = mean_squared_error(y_test, y_pred)
    # RMSE
    rmse = np.sqrt(mse)
    # R2
    r2 = r2_score(y_test, y_pred)
   

    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2:', r2)
