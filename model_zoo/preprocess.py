from sklearn.preprocessing import StandardScaler


def standardize(fit_data, extra_data=None):
    """
    standardize data
    :param fit_data: data to fit and transform
    :param extra_data: extra data to transform
    :return:
    """
    s = StandardScaler()
    s.fit(fit_data)
    fit_data = s.transform(fit_data)
    if not extra_data is None:
        extra_data = s.transform(extra_data)
        return fit_data, extra_data
    return fit_data
